import torch
from torch import Tensor
import dgl
import logging

class History(torch.nn.Module):
    """A historical embedding storage module"""
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()

        self.ratio = 0.4

        device = torch.device('cuda')
 
        self.num_nodes = num_nodes
        self.num_embeddings = int(num_nodes * self.ratio)
        self.embedding_dim = embedding_dim

        self.emb = torch.empty(self.num_embeddings, embedding_dim, device=device)

        self.pos = torch.empty(self.num_nodes, device = device, dtype=torch.long)           # 指示顶点特征所在的历史嵌入索引

        self.index_to_gid = torch.empty(self.num_embeddings, device = device, dtype = torch.long)

        self.index = 0

        self.reset_parameters()
    
    def reset_parameters(self):
        self.emb.fill_(0)
        self.pos.fill_(-1)
        self.index_to_gid.fill_(-1)
        self.index = 0

    '''
    def push(self, gids, feats, grad, grad_thresh):
        #环形缓冲区
        dim = self.embedding_dim
        grad_stat = torch.norm(grad, dim=1)
        lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long).tolist()    # 所有满足梯度阈值条件的下标
        # lid = [b for a in lid for b in a]   
        gid = [gids[id] for id in lid]
        for i in range(len(gid)):
            g_id = gid[i]
            l_id = lid[i]
            if(self.pos[g_id] == -1):
                index = int(self.index % self.num_embeddings)
                if(self.index_to_gid[index] != -1 and self.pos[self.index_to_gid[index]] != -1):
                    self.pos[self.index_to_gid[index]] = -1
                self.emb[index] = feats[l_id].detach()
                self.pos[g_id] = index
                self.index_to_gid[index] = g_id
                self.index += 1
    '''

    def push(self, gids, feats, grad, grad_thresh):
        '''环形缓冲区'''
        if self.num_embeddings == 0:
            return
        dim = self.embedding_dim
        grad_stat = torch.norm(grad, dim=1)
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

    '''
    def push(self, gids, feats, grad, grad_thresh):
        """固定缓冲区"""
        dim = self.embedding_dim
        grad_stat = torch.norm(grad, dim=1)
        lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long).tolist()    # 所有满足梯度阈值条件的下标
        # lid = [b for a in lid for b in a]   
        gid = [gids[id] for id in lid]
        #print("before: ", self.index)
        for index in range(len(gid)):
            g_id = gid[index]
            l_id = lid[index]
            if((self.index < self.num_embeddings) and (self.pos[g_id] == -1)):
                self.emb[self.index] = feats[l_id].detach()
                self.pos[g_id] = self.index
                self.index += 1
        
            # if((self.index < self.num_embeddings) and (torch.any(self.pos.eq(g_id))==False)):
            #     self.emb[self.index] = feats[l_id].detach()
            #     self.pos[self.index] = g_id
            #     self.index += 1
            #else:
            #    return
        #print("after: ", self.index, "num: ", len(lid))
    '''

    '''
    def push(self, gids, feats, grad, grad_thresh):
        """固定缓冲区"""
        dim = self.embedding_dim
        grad_stat = torch.norm(grad, dim=1)
        lid = grad_stat.le(grad_thresh).nonzero().squeeze().type(torch.long).tolist()    # 所有满足梯度阈值条件的下标
        # lid = [b for a in lid for b in a]   
        gid = [gids[id] for id in lid]
        #print("before: ", self.index)
        for index in range(len(gid)):
            g_id = gid[index]
            l_id = lid[index]
            if((self.index < self.num_embeddings) and (self.pos[g_id] == -1)):
                self.emb[self.index] = feats[l_id].detach()
                self.pos[g_id] = self.index
                self.index += 1
    '''

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