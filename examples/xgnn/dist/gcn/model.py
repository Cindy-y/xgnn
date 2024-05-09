import dgl
import torch.nn as nn

from xgnn.nn import GCNConv
import xgnn.distributed as dist
from xgnn.distributed.history import History


class ModelParallelStage(nn.Module):
    def __init__(self, in_feats, n_hidden, n_workers):
        """
        1. in_feats的大小是切分后的特征维度
        2. 模型并行阶段不能执行激活函数
        """
        super().__init__()
        self.n_workers = n_workers
        self.model_layer = GCNConv(in_feats, n_hidden, allow_zero_in_degree=True)

    def forward(self, blocks, feats):
        output = []
        for i in range(self.n_workers):
            assert (
                blocks[i].srcnodes().shape[0] == feats[i].shape[0]
            ), "number of nodes {} is not equal to the number of features {}!".format(
                blocks[i].srcnodes().shape[0], feats[i].shape[0]
            )
            x = self.model_layer(blocks[i], feats[i])
            output.append(x)
        return output


class DataParallelStage(nn.Module):
    """数据并行阶段"""

    def __init__(self, num_nodes, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.num_nodes =  num_nodes
        self.n_layers = n_layers           # 2
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        self.feat_grad = []
        self.feat = []
        self.histories = nn.ModuleList([
            History(num_nodes, n_hidden) for _ in range(n_layers)
        ])
        for _ in range(1, n_layers):
            self.layers.append(
                GCNConv(
                    n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True
                )
            )
            self.feat_grad.append({})
        self.layers.append(
            GCNConv(
                n_hidden, n_classes, activation=activation, allow_zero_in_degree=True
            )
        )
        self.feat_grad.append({})
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):

        # self.feat.clear()
        
        #for i in range(0, self.n_layers):
        #    self.feat_grad[i] = {}
        def save_grad(i, name):
            def hook(grad):
                self.feat_grad[i][name] = grad
                # print("features grad shape: {}".format(grad.shape))

            return hook
        # 数据并行
        #self.histories[0].d_pull(blocks[0].srcdata[dgl.NID], blocks[0].srcnodes(), x)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            assert (
                block.srcnodes().shape[0] == x.shape[0]
            ), "number of nodes {} is not equal to the number of features {}!".format(
                block.srcnodes().shape[0], x.shape[0]
            )
            
            if(l == self.n_layers-1):
                x = layer(block, x)
            else:
                history = self.histories[l+1]
                x = layer(block, x) 
                x.register_hook(save_grad(l+1, "feats"))   
                self.feat.append(x)  
                history.pull(block, x)            
                x = self.dropout(x)      

        return x


class DistGCN(dist.StageModule):
    def __init__(
        self,
        num_nodes,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        n_workers,
        device,
        activation,
        dropout,
    ):
        # 模型并行
        model_stage = ModelParallelStage(in_feats, n_hidden, n_workers).to(device)
        # 数据并行
        data_stage = DataParallelStage(
            num_nodes,n_hidden, n_classes, n_layers - 1, activation, dropout
        ).to(device)
        super().__init__(model_stage, data_stage, device, activation, dropout)
