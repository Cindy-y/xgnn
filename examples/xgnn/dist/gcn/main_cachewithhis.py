import argparse
import json
import logging
import os
import socket
import time

import dgl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import register_data_args
from dgl.data.utils import load_tensors

import xgnn.distributed.optimizer as optimizer
from xgnn.distributed.communication import pull_subgraph
from xgnn.distributed.runtime import StageRuntime

from xgnn.cache import CSRTopo, Feature, init_p2p

from model import DistGCN

SAMPLE_TIME = 0
SAMPLE_COMM_TIME = 0
FEATURE_TIME = 0
PRUNE_TIME = 0



class NeighborSampler:
    def __init__(self, g, node_feats, fanouts, world_size, device, history):
        self.g = g
        # self.node_feats = node_feats
        self.fanouts = fanouts
        self.device = device
        self.world_size = world_size
        self.history = history

    # 采样器进程调用用来执行分布式采样
    def sample(self, seeds):
        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        global SAMPLE_TIME
        global PRUNE_TIME
        s = time.time()
        # 采样
        histories = self.history

        for l,fanout in enumerate(self.fanouts):                              
            frontier = dgl.distributed.sample_neighbors(
                self.g, seeds, fanout, replace=True
            )
            block = dgl.to_block(frontier, seeds)
            
            if(l==0):
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            elif(l==(len(self.fanouts)-1)):
                blocks[0].srcdata["pruned"] = torch.zeros(block.dstdata[dgl.NID].shape[0], dtype=torch.bool)
                block.dstdata["pruned"] = torch.zeros(block.dstdata[dgl.NID].shape[0], dtype=torch.bool)
                # print("block1: ", block)
                s1 = time.time()
                block = block.to(self.device)
                history = histories[len(histories)-l]
                # s1 = time.time()
                block = history.prune(block).to(torch.device('cpu'))
                blocks[0].srcdata["pruned"] = block.dstdata["pruned"]
                PRUNE_TIME += time.time() - s1
                # print("block2: ", block)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            else:
                block.dstdata["pruned"] = torch.zeros(block.dstdata[dgl.NID].shape[0], dtype=torch.bool)
                # print("block1: ", block)
                s1 = time.time()
                block = block.to(self.device)
                history = histories[len(histories)-l]
                # s1 = time.time()
                block = history.prune(block).to(torch.device('cpu'))
                PRUNE_TIME += time.time() - s1
                # print("block2: ", block)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            '''
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
            '''
        SAMPLE_TIME += time.time() - s
        seeds = blocks[-1].dstdata[dgl.NID]
        blocks[-1].dstdata["labels"] = self.g.ndata["labels"][seeds]
        return blocks


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def run(args, device, data, in_feats, num_nodes, dev_id):
    rank = int(os.environ["RANK"])
    print("device {}: rank: {} in run :".format(device, rank))
    world_size = int(os.environ["WORLD_SIZE"])
    print("rank: {}, world_size: {}".format(rank, world_size))
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data
    print(
        "rank: {}, train size: {}, val size: {}, test size: {}".format(
            rank, train_nid.shape[0], val_nid.shape[0], test_nid.shape[0]
        )
    )

    model = DistGCN(
        num_nodes,
        in_feats.shape[-1],
        args.num_hidden,
        args.n_classes,
        args.num_layers,
        world_size,
        device,
        F.relu,
        args.dropout,
        args.ratio,
        args.pgrad,
        args.tstale,
    )

    shuffle = True
    # Create sampler
    sampler = NeighborSampler(
        g,
        in_feats,
        [int(fanout) for fanout in args.fan_out.split(",")],
        world_size,
        device,
        model.data_stage.module.histories
    )

    # Create DataLoader for constructing blocks
    dataloader = dgl.distributed.DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample,
        shuffle=shuffle,
        drop_last=False,
    )
    val_dataloader = dgl.distributed.DistDataLoader(
        dataset=val_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample,
        shuffle=shuffle,
        drop_last=False,
    )
    test_dataloader = dgl.distributed.DistDataLoader(
        dataset=test_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample,
        shuffle=shuffle,
        drop_last=False,
    )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    param_version = args.num_warmup + 1
    optim = optimizer.AdamWithWeightStashing(model, param_version, args.lr)

    runtime = StageRuntime(
        model, in_feats, rank, world_size, device, optim, loss_fcn, num_warmup=args.num_warmup
    )
    all_time = []
    all_only_sample_time = []
    all_prune__time = []
    all_sample_comm_time = []
    all_feature_time = []
    accumulate_time = []
    accuracy_list = []
    zero_time = time.time()
    useHistory = True
    draw_accu_time = []
    draw_acc = []
    begin_time = time.time()
    runtime.model.data_stage.module.start_history()
    for epoch in range(args.num_epochs):
        start = time.time()
        global SAMPLE_TIME, SAMPLE_COMM_TIME, FEATURE_TIME, PRUNE_TIME
        SAMPLE_TIME, SAMPLE_COMM_TIME, FEATURE_TIME, PRUNE_TIME = 0, 0, 0, 0
        num_iter = (train_nid.shape[-1] + args.batch_size) // args.batch_size
        # 训练
        runtime.train(num_iter, dataloader)
        # warm_up
        runtime.forward()                                                         
        for it in range(num_iter):
            output, loss, target = runtime.forward()
            runtime.backward_and_step()
            accuracy_list.append(compute_acc(output, target).item())
            draw_accu_time.append(time.time() - begin_time)
            draw_acc.append(compute_acc(output, target).item())
            if it % 20 == 0:
                # if rank == 0:
                acc = compute_acc(output, target)
                mem = torch.cuda.max_memory_allocated() / 1000000
                ctime = time.time() - zero_time
                accumulate_time.append(ctime)
                print(
                    "Epoch: {}, step: {}, loss: {}, acc: {}, GPU mem: {}, cumulative time: {}".format(
                        # "{},{},{},{},{},{}".format(
                        epoch,
                        it,
                        loss,
                        acc.item(),
                        mem,
                        ctime,
                    )
                )
        torch.cuda.synchronize()
        all_time.append(time.time() - start)
        all_only_sample_time.append(SAMPLE_TIME)
        all_prune__time.append(PRUNE_TIME)
        all_sample_comm_time.append(SAMPLE_COMM_TIME)
        all_feature_time.append(FEATURE_TIME)
        """
        if useHistory:
            accuracy_ascend = [(b-a)/b for a,b in zip(accuracy_list, accuracy_list[1:])]
            if sum(accuracy_ascend) < 0.5:
                useHistory = False
                #runtime.model.data_stage.module.change_history(0.3, 100)
        """
        accuracy_list = []
        """"""
        print(
            "Epoch: {}, iteration nums: {}, time: {} (S), sample time:{}, sample comm time: {}, feature time: {}, prune time: {}".format(
                epoch,
                it,
                time.time() - start,
                SAMPLE_TIME,
                SAMPLE_COMM_TIME,
                FEATURE_TIME,
                PRUNE_TIME,
            )
        )
        # 验证集
        if args.eval and epoch % args.eval_every == 0 and epoch != 0:
            num_iter = (val_nid.shape[-1] + args.batch_size) // args.batch_size
            runtime.eval(num_iter, val_dataloader)
            start = time.time()
            acc = evaluate(runtime, num_iter)
            mem = torch.cuda.max_memory_allocated() / 1000000
            eval_time = time.time() - start
            print(
                "Epoch: {}, val acc: {}, Avg time: {}, GPU mem: {}".format(
                    epoch, acc.item(), eval_time, mem
                )
            )
    mean_time = torch.tensor(all_time, dtype=torch.float32).mean()
    mean_only_sample_time = torch.tensor(
        all_only_sample_time, dtype=torch.float32
    ).mean()
    mean_prune_time = torch.tensor(
        all_prune__time, dtype = torch.float32
    ).mean()
    mean_sample_comm_time = torch.tensor(
        all_sample_comm_time, dtype=torch.float32
    ).mean()
    mean_feature_time = torch.tensor(all_feature_time, dtype=torch.float32).mean()
    print(
        "Epoch mean time: {} s, mean sample time: {}, mean sample comm time: {}, mean feature time: {}, mean prune time: {}".format(
            mean_time,
            mean_only_sample_time,
            mean_sample_comm_time,
            mean_feature_time,
            mean_prune_time,
        )
    )
    # if rank == 0:
    #     for (accu_time, acc) in zip(draw_accu_time, draw_acc):
    #         print("[", accu_time, ", ", acc, "],")
    if args.eval:
        # 测试集
        num_iter = (test_nid.shape[-1] + args.batch_size) // args.batch_size
        runtime.eval(num_iter, test_dataloader)
        start = time.time()
        acc = evaluate(runtime, num_iter)
        mem = torch.cuda.max_memory_allocated() / 1000000
        eval_time = time.time() - start
        print(
            "Epoch: {}, test acc: {}, Avg time: {}, GPU mem: {}".format(
                epoch, acc.item(), eval_time, mem
            )
        )


def evaluate(runtime, num_iter):
    all_labels = []
    all_output = []
    with torch.no_grad():
        for _ in range(num_iter):
            output, loss, target = runtime.forward()
            all_labels.append(target)
            all_output.append(output)
    label = torch.cat(all_labels, 0)
    output = torch.cat(all_output, 0)
    acc = compute_acc(output, label)
    return acc


def load_features(part_config, part_id):
    config_path = os.path.dirname(part_config)
    relative_to_config = lambda path: os.path.join(config_path, path)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert "part-{}".format(part_id) in part_metadata, "part-{} does not exist".format(
        part_id
    )
    part_files = part_metadata["part-{}".format(part_id)]
    assert "part_feats" in part_files, "the partition does not contain node features."
    part_feats = load_tensors(relative_to_config(part_files["part_feats"]))
    print("part id: {}, feats shape: {}".format(part_id, part_feats["feat"].shape))
    return part_feats["feat"]


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        dist.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(socket.gethostname(), "rank:", g.rank())

    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(g.ndata["val_mask"], pb, force_even=True)
        test_nid = dgl.distributed.node_split(g.ndata["test_mask"], pb, force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    if args.num_gpus == -1:
        device = torch.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = torch.device("cuda:" + str(dev_id))
    print("Part:{}, device: {}".format(g.rank(), str(device)))
    labels = g.ndata["labels"][np.arange(g.number_of_nodes())]
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    print("#labels:{}, label shape: {}".format(n_classes, g.ndata["labels"].shape))

    num_nodes = g.ndata["labels"].shape[0]

    rank = int(os.environ["RANK"])

    in_feats = load_features(args.part_config, rank)

    csr_topo = CSRTopo(torch.stack(g.local_partition.edges('uv')))
    # init_p2p([0,1])
    print("CSRTopo construct succeed!")
    nfeat = Feature(rank = 0, device_list=[0],
                    device_cache_size="200M", cache_policy="device_replicate",
                    csr_topo=csr_topo)
    print("Feature construct succeed!")
    nfeat.from_cpu_tensor(in_feats)
    print("nfeat get!")

    data = train_nid, val_nid, test_nid, n_classes, g
    run(args, device, data, nfeat, num_nodes, dev_id)
    # run(args, device, data, in_feats, num_nodes, dev_id)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARN)
    parser = argparse.ArgumentParser(description="GCN")
    register_data_args(parser)
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config", type=str, help="The file for IP configuration")
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument("--num_clients", type=int, help="The number of clients")
    parser.add_argument(
        "--n_classes", type=int, default=47, help="the number of classes"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="pytorch distributed backend"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--pgrad", type=float, default=0.9)
    parser.add_argument("--tstale", type=int, default=25)
    parser.add_argument("--num_samplers", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fan_out", type=str, default="25,10,10")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval", action="store_true", help="evaluate the model")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--local_rank", type=int, help="get rank of the process")
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument("--num_warmup", type=int, help="number of warmup", default=2)
    args = parser.parse_args()

    print(args)
    main(args)
