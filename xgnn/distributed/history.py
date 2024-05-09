import torch
from torch import Tensor
import dgl
import logging

class History(torch.nn.Module):
    """A historical embedding storage module"""
    def __init__(self, num_nodes: int, embedding_dim: int, ratio, pgrad, tstale, device):
        super().__init__()

        self.ratio = ratio
        self.pgrad = pgrad
        self.tstale = int(tstale)

        self.device = device
 
        self.num_nodes = num_nodes
        self.num_embeddings = 0
        self.embedding_dim = embedding_dim

        self.pos = torch.empty(self.num_nodes, device = self.device, dtype=torch.long)           # 指示顶点特征所在的历史嵌入索引
        self.pos.fill_(-1)
        self.time = torch.empty(self.num_nodes, device = self.device, dtype=torch.int)
        self.time.fill_(0)

    def set_pgrad(self, pgrad):
        self.pgrad = pgrad
    
    def set_tstale(self, tstale):
        self.tstale = tstale

    def start(self):
        self.num_embeddings = int(100)
        # print("self.num_embeddings: ", self.num_embeddings)
        self.emb = torch.empty(self.num_embeddings, self.embedding_dim, device=self.device)
        self.index_to_gid = torch.empty(self.num_embeddings, device = self.device, dtype = torch.long)
        self.emb.fill_(0)
        self.index_to_gid.fill_(-1)

    def __init__20(self, num_nodes: int, embedding_dim: int, ratio, pgrad, tstale):
        super().__init__()

        self.ratio = ratio
        self.pgrad = pgrad
        self.tstale = int(tstale)

        self.device = torch.device('cuda')
 
        self.num_nodes = num_nodes
        self.num_embeddings = 0
        self.embedding_dim = embedding_dim

        self.pos = torch.empty(self.num_nodes, device = self.device, dtype=torch.long)           # 指示顶点特征所在的历史嵌入索引
        self.pos.fill_(-1)
        self.time = torch.empty(self.num_nodes, device = self.device, dtype=torch.int)
        self.time.fill_(0)

        # self.index = 0
        self.embInit = False

    def __init__10(self, num_nodes: int, embedding_dim: int, ratio, pgrad, tstale):
        super().__init__()

        self.ratio = ratio
        self.pgrad = pgrad
        self.tstale = int(tstale)

        self.device = torch.device('cuda')
 
        self.num_nodes = num_nodes
        self.num_embeddings = 0
        self.embedding_dim = embedding_dim

        self.emb = torch.tensor([], device=self.device)

        self.pos = torch.empty(self.num_nodes, device = self.device, dtype=torch.long)           # 指示顶点特征所在的历史嵌入索引

        self.index_to_gid = torch.tensor([], device = self.device, dtype = torch.long)

        self.index = 0

        self.pos.fill_(-1)

        # self.reset_parameters()

    def __init__1(self, num_nodes: int, embedding_dim: int, ratio, pgrad, tstale):
        super().__init__()

        self.ratio = ratio
        self.pgrad = pgrad
        self.tstale = int(tstale)

        self.device = torch.device('cuda')
 
        self.num_nodes = num_nodes
        self.num_embeddings = int(num_nodes * self.ratio)
        self.embedding_dim = embedding_dim

        self.emb = torch.empty(self.num_embeddings, embedding_dim, device=self.device)

        self.pos = torch.empty(self.num_nodes, device = self.device, dtype=torch.long)           # 指示顶点特征所在的历史嵌入索引

        self.index_to_gid = torch.empty(self.num_embeddings, device = self.device, dtype = torch.long)

        self.index = 0

        self.reset_parameters()
    
    def reset_parameters(self):
        self.emb.fill_(0)
        self.pos.fill_(-1)
        self.index_to_gid.fill_(-1)
        self.index = 0

    def emb_size(self):
        print("emb_size: {}".format(self.emb.size()))

    def push(self, gids, feats, grad):
        '''
        过期时间缓冲区: 
        缓冲区大小初始化为一个非0的较小数,比如100
        梯度满足pgrad阈值比例的中间嵌入缓存在缓冲区中:
            如果缓冲区空位足够,则直接缓存
            空位不够,则直接增大缓存,在尾部增加相应的空位
        每一次push结束,非空位置的缓存时间就加1,然后移除缓存时间超过tstale的位置
        '''
        if self.num_embeddings == 0:
            return

        resetIndexs = self.pos[gids]
        self.pos[gids] = -1
        self.index_to_gid[resetIndexs] = -1
        self.time[gids] = 0

        grad_stat = torch.norm(grad, dim=1)
        point = (int)(grad.shape[0] * self.pgrad)
        grad_thresh = torch.kthvalue(grad_stat, point).values.squeeze()
        lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long)    # 所有满足梯度阈值条件的下标
        gid = gids[lid]
        num = gid.shape[0]
        if num == 0:
            return

        poses = self.index_to_gid.eq(-1).nonzero().squeeze().type(torch.long) # 空着的位置
        if(poses.shape[0] < num):
            emb = torch.empty(num - poses.shape[0], self.embedding_dim, device = self.device)
            emb.fill_(0)
            index_to_gid = torch.empty(num - poses.shape[0], device = self.device, dtype=torch.long)
            index_to_gid.fill_(-1)

            self.emb = torch.cat([self.emb, emb], dim=0)
            self.index_to_gid = torch.cat([self.index_to_gid, index_to_gid])

            self.num_embeddings += (num - poses.shape[0])
            # print("self.num_embeddings: ", self.num_embeddings)
        poses = self.index_to_gid.eq(-1).nonzero().squeeze().type(torch.long)
        poses = poses[0:num]
        self.emb[poses] = feats[lid].detach()
        self.pos[gid] = poses
        self.index_to_gid[poses] = gid

        indexs = self.pos.ne(-1).nonzero().squeeze().type(torch.long) 
        self.time[indexs] = self.time[indexs] + 1
        gid = self.time.ge(self.tstale).nonzero().squeeze().type(torch.long) 
        resetIndexs = self.pos[gid]
        self.pos[gid] = -1
        self.index_to_gid[resetIndexs] = -1
        self.time[gid] = 0


    def push20(self, gids, feats, grad):      
        '''
        过期时间缓冲区: 
        缓冲区大小初始化为 第一次push的顶点总数 * pgrad * tstale
        梯度满足pgrad阈值比例的中间嵌入缓存在缓冲区中:
            如果缓冲区空位足够,则直接缓存
            空位不够,则移除缓存时间最久的位置,并把该位置时间重置为0
        每一次push结束,非空位置的缓存时间就加1
        '''
        if not self.embInit:
            self.num_embeddings = int(grad.shape[0] * self.pgrad * self.tstale)
            #print("self.num_embeddings: ", self.num_embeddings)
            self.emb = torch.empty(self.num_embeddings, self.embedding_dim, device=self.device)
            self.index_to_gid = torch.empty(self.num_embeddings, device = self.device, dtype = torch.long)
            self.emb.fill_(0)
            self.index_to_gid.fill_(-1)
            self.embInit = True

        if self.num_embeddings == 0:
            return

        resetIndexs = self.pos[gids]
        self.pos[gids] = -1
        self.index_to_gid[resetIndexs] = -1
        self.time[gids] = 0

        grad_stat = torch.norm(grad, dim=1)
        point = (int)(grad.shape[0] * self.pgrad)
        grad_thresh = torch.kthvalue(grad_stat, point).values.squeeze()
        lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long)    # 所有满足梯度阈值条件的下标
        gid = gids[lid]
        num = gid.shape[0]
        if num == 0:
            return

        poses = self.index_to_gid.eq(-1).nonzero().squeeze().type(torch.long) # 空着的位置
        if(poses.shape[0] < num):
            delete_time = torch.kthvalue(self.time, self.num_nodes - (num - poses.shape[0])).values.squeeze()
            deleteLid = self.time.ge(delete_time).nonzero().squeeze().type(torch.long) 
            deleteGid = gids[deleteLid]
            deleteIndex = self.pos[deleteGid]
            self.pos[deleteGid] = -1
            self.index_to_gid[deleteIndex] = -1
            self.time[deleteGid] = 0
            
        poses = self.index_to_gid.eq(-1).nonzero().squeeze().type(torch.long)
        poses = poses[0:num]
        self.emb[poses] = feats[lid].detach()
        self.pos[gid] = poses
        self.index_to_gid[poses] = gid

        indexs = self.pos.ne(-1).nonzero().squeeze().type(torch.long) 
        self.time[indexs] = self.time[indexs] + 1
        '''
        gid = self.time.ge(self.tstale).nonzero().squeeze().type(torch.long) 
        resetIndexs = self.pos[gid]
        self.pos[gid] = -1
        self.index_to_gid[resetIndexs] = -1
        self.time[gid] = 0
        '''


    def push1(self, gids, feats, grad):
        # self.emb_size()
        
        '''
        环形缓冲区: 
        缓冲区大小初始化为num_nodes * ratio
        梯度满足pgrad阈值比例的中间嵌入缓存在缓冲区中,到达尾部的部分覆盖缓冲区的头部;同时不满足的将缓冲区的有效位置为-1
        '''
        pgrad = self.pgrad
        if self.num_embeddings == 0:
            return
        dim = self.embedding_dim
        grad_stat = torch.norm(grad, dim=1)
        point = (int)(grad_stat.shape[0] * pgrad)
        grad_thresh = torch.kthvalue(grad_stat, point).values.squeeze()
        # sorted_grad_stat = torch.sort(grad_stat, dim=0)
        # sorted_values = sorted_grad_stat.values
        # sorted_indices = sorted_grad_stat.indices
        deleteLid = grad_stat.ge(grad_thresh).nonzero().squeeze().type(torch.long) 
        deleteGid = gids[deleteLid]
        deleteIndex = self.pos[deleteGid]
        self.pos[deleteGid] = -1
        self.index_to_gid[deleteIndex] = -1

        lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long)    # 所有满足梯度阈值条件的下标
        gid = gids[lid]
        num = gid.shape[0]
        if num == 0:
            return
        self.index = self.index % self.num_embeddings
        numTail = self.num_embeddings - self.index
        numHead = 0
        if(numTail >= num):
            numTail = num
        else:
            numHead = num - numTail
        indexTail = torch.arange(self.index, self.index + numTail, 1, device='cuda')
        indexHead = torch.arange(0, numHead, 1, device='cuda')
        indexs = torch.cat((indexTail, indexHead), dim = 0)

        indexsToGids = self.index_to_gid[indexs]
        usedIndexs = indexsToGids.ne(-1).nonzero().squeeze().type(torch.long)
        usedGids = indexsToGids[usedIndexs]
        self.pos[usedGids] = -1
        self.index_to_gid[indexs] = -1

        self.emb[indexs] = feats[lid].detach()
        self.pos[gid] = indexs
        self.index_to_gid[indexs] = gid
        self.index += num


    def push10(self, gids, feats, grad):
        '''
        环形缓冲区: 
        缓冲区大小初始化为0
        前tstale次push都直接增大缓冲区,将梯度满足pgrad阈值比例的中间嵌入缓存在缓冲区尾部
        tstale次之后,梯度满足pgrad阈值比例的中间嵌入缓存在缓冲区中,到达尾部的部分覆盖缓冲区的头部;同时不满足的将缓冲区的有效位置为-1
        '''

        pgrad = self.pgrad
        tstale = self.tstale

        if(self.tstale > 0):

            self.tstale -= 1
        
            grad_stat = torch.norm(grad, dim=1)
            point = (int)(grad_stat.shape[0] * pgrad)
            grad_thresh = torch.kthvalue(grad_stat, point).values.squeeze()
            lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long)    # 所有满足梯度阈值条件的下标
            gid = gids[lid]
            num = gid.shape[0]

            emb = torch.empty(num, self.embedding_dim, device = self.device)
            emb.fill_(0)
            index_to_gid = torch.empty(num, device = self.device, dtype=torch.long)
            index_to_gid.fill_(-1)

            self.emb = torch.cat([self.emb, emb], dim=0)
            self.index_to_gid = torch.cat([self.index_to_gid, index_to_gid])

            self.num_embeddings += num

            indexs = torch.arange(self.index, self.index + num, 1, device='cuda')
            self.emb[indexs] = feats[lid].detach()
            self.pos[gid] = indexs
            self.index_to_gid[indexs] = gid
            self.index += num

        if(self.tstale == 0):

            # self.emb_size()

            if self.num_embeddings == 0:
                return
            grad_stat = torch.norm(grad, dim=1)
            point = (int)(grad_stat.shape[0] * pgrad)
            grad_thresh = torch.kthvalue(grad_stat, point).values.squeeze()
            # sorted_grad_stat = torch.sort(grad_stat, dim=0)
            # sorted_values = sorted_grad_stat.values
            # sorted_indices = sorted_grad_stat.indices
            deleteLid = grad_stat.ge(grad_thresh).nonzero().squeeze().type(torch.long) 
            deleteGid = gids[deleteLid]
            deleteIndex = self.pos[deleteGid]
            self.pos[deleteGid] = -1
            self.index_to_gid[deleteIndex] = -1

            lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long)    # 所有满足梯度阈值条件的下标
            gid = gids[lid]
            num = gid.shape[0]
            if num == 0:
                return
            self.index = self.index % self.num_embeddings
            numTail = self.num_embeddings - self.index
            numHead = 0
            if(numTail >= num):
                numTail = num
            else:
                numHead = num - numTail
            indexTail = torch.arange(self.index, self.index + numTail, 1, device='cuda')
            indexHead = torch.arange(0, numHead, 1, device='cuda')
            indexs = torch.cat((indexTail, indexHead), dim = 0)

            indexsToGids = self.index_to_gid[indexs]
            usedIndexs = indexsToGids.ne(-1).nonzero().squeeze().type(torch.long)
            usedGids = indexsToGids[usedIndexs]
            self.pos[usedGids] = -1
            self.index_to_gid[indexs] = -1

            self.emb[indexs] = feats[lid].detach()
            self.pos[gid] = indexs
            self.index_to_gid[indexs] = gid
            self.index += num


    def prune(self, block):
        dst_gid = block.dstdata[dgl.NID]
        poses = self.pos[dst_gid].ne(-1).nonzero().squeeze().type(torch.long)
        if poses.shape[0] == 0:
            return block
        lids = block.dstnodes()[poses]
        block.dstdata["pruned"][lids] = True
        eids = block.in_edges(lids, 'eid')
        block.remove_edges(eids)

        src_nodes = block.srcdata[dgl.NID]
        src = src_nodes[block.edges()[0]]
        dst = src_nodes[block.edges()[1]]
        dst_nodes = block.dstdata[dgl.NID]
        block2 = dgl.to_block(dgl.graph((src, dst)), dst_nodes=dst_nodes)
        block2.dstdata["pruned"] = block.dstdata["pruned"]

        return block2
        
    def pull(self, block, feats):
        '''使用历史嵌入更新特征向量'''
        lids = block.dstdata["pruned"].nonzero().squeeze().type(torch.long)
        if lids.shape[0] != 0:
            gids = block.dstdata[dgl.NID][lids]
            indexs = self.pos[gids]
            feats[lids][:] = self.emb[indexs]
            # feats[lids].detach()
            
            # lids = lids.tolist()
            # gids = gids.tolist()
            # for i in range(len(gids)):
            #     gid = gids[i]
            #     lid = lids[i]
            #     index = self.pos.eq(gid).nonzero().squeeze().type(torch.long).tolist()
            #     feats[lids[i]][:] = self.emb[index]
            #     # feats[lids[i]].detach()

    def d_pull(self, block, feats):
        '''使用历史嵌入更新特征向量'''
        lids = block.srcdata["pruned"].nonzero().squeeze().type(torch.long)
        if lids.shape[0] != 0:
            gids = block.srcdata[dgl.NID][lids]
            indexs = self.pos[gids]
            feats[lids][:] = self.emb[indexs]    
        

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError


    '''
    def prune(self, block):
        if torch.all(self.pos.eq(-1)):
            return block
        else:
            src_gid = block.srcdata[dgl.NID].tolist()
            dst_gid = block.dstdata[dgl.NID].tolist()
            src_lid = block.srcnodes().tolist()
            dst_lid = block.dstnodes().tolist()
            edges = block.all_edges()
            eids = []
            src,dst = edges[0], edges[1]
            for i in range(len(dst_lid)):
                lid = dst_lid[i]
                gid = dst_gid[i] 
                if(self.pos[gid] != -1):
                    index = dst.eq(lid).nonzero().squeeze().type(torch.long).tolist()
                    block.dstdata["pruned"][lid] = True
                    eids.append(index)

                # if(torch.any(self.pos.eq(gid))==True):
                #     index = dst.eq(lid).nonzero().squeeze().type(torch.long).tolist()
                #     block.dstdata["pruned"][lid] = True
                #     eids.append(index)
            eids = [b for a in eids for b in a]
            block.remove_edges(eids)

            #out_degrees = block.out_degrees()
            #index1 = out_degrees.eq(0).nonzero().squeeze().type(torch.long).tolist()
            #nids = list(set(index1) - set(dst_lid))
            #block.remove_nodes(nids)

            src_nodes = block.srcdata[dgl.NID]
            src = src_nodes[block.edges()[0]]
            dst = src_nodes[block.edges()[1]]
            dst_nodes = block.dstdata[dgl.NID]
            block2 = dgl.to_block(dgl.graph((src, dst)), dst_nodes=dst_nodes)
            block2.dstdata["pruned"] = block.dstdata["pruned"]
            
            return block2
    '''