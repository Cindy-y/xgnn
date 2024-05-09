from scipy.sparse import csr_matrix
import numpy as np
import torch
from typing import List
from xgnn import propeller

def find_cliques(adj_mat, clique_res, remaining_nodes, potential_clique,
                 skip_nodes):
    """
    寻找最大完全子图：满足任意两点都恰有一条边相连的子图，也叫团
    """
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        clique_res.append(potential_clique)
        return 1

    found_cliques = 0
    for node in remaining_nodes:

        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [
            n for n in remaining_nodes if adj_mat[node][n] == 1
        ]
        new_skip_list = [n for n in skip_nodes if adj_mat[node][n] == 1]

        found_cliques += find_cliques(adj_mat, clique_res, new_remaining_nodes,
                                      new_potential_clique, new_skip_list)

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)
    return found_cliques


def color_mat(access_book, device_list):
    device2clique = dict.fromkeys(device_list, -1)
    clique2device = {}
    clique_res = []
    all_nodes = list(range(len(device_list)))
    if len(device_list) == 8:
        clique_res = [[0,1,2,3],[4,5,6,7]]
    else:
        find_cliques(access_book, clique_res, all_nodes, [], [])
    for index, clique in enumerate(clique_res):
        clique2device[index] = []
        for device_idx in clique:
            clique2device[index].append(device_list[device_idx])       ## 子图index到所有相连设备号的映射 : 以index为键，设备号为值的字典
            device2clique[device_list[device_idx]] = index             ### 设备号到子图index的映射 ： 以设备号为键，index为值

    return device2clique, clique2device


class Topo:
    """P2P access topology for devices. Normally we use this class to detect the connection topology of GPUs on the machine.
    
    ```python
    >>> p2p_clique_topo = p2pCliqueTopo([0,1])
    >>> print(p2p_clique_topo.info())
    ```

    Args:
        device_list ([int]): device list for detecting p2p access topology
        
    
    """
    def __init__(self, device_list: List[int]) -> None:
        access_book = torch.zeros((len(device_list), len(device_list)))
        for src_index, src_device in enumerate(device_list):
            for dst_index, dst_device in enumerate(device_list):
                if src_index != dst_index and propeller.can_device_access_peer(
                        src_device, dst_device):
                    access_book[src_index][dst_index] = 1
                    access_book[dst_index][src_index] = 1
        self.Device2p2pClique, self.p2pClique2Device = color_mat(
            access_book, device_list)

    def get_clique_id(self, device_id: int):
        """Get clique id for device with device_id 

        Args:
            device_id (int): device id of the device

        Returns:
            int: clique_id of the device
        """
        return self.Device2p2pClique[device_id]

    def info(self):
        """Get string description for p2p access topology, you can call `info()` to check the topology of your GPUs 

        Returns:
            str: p2p access topology for devices in device list
        """
        str = ""
        for clique_idx in self.p2pClique2Device:
            str += f"Devices {self.p2pClique2Device[clique_idx]} support p2p access with each other\n"
        return str

    @property
    def p2p_clique(self):
        """get all p2p_cliques constructed from devices in device_list

        Returns:
            Dict : {clique_id:[devices in this clique]}
        """
        return self.p2pClique2Device
    
def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix((data, (src, dst)))
    return csr_mat

class CSRTopo:
    """Graph topology in CSR format.
    Args:
        edge_index ([torch.longTensor], optional): edge_index tensor for graph topo
        indptr (torch.LongTensor, optional): indptr for CSR format graph topo
        indices (torch.LongTensor, optional): indices for CSR format graph topo
    """
    def __init__(self, edge_index=None, indptr=None, indices = None, eid=None):
        print("CSRTopo init...")
        if edge_index is not None:
            csr_mat = get_csr_from_coo(edge_index)
            self.indptr_ = torch.from_numpy(csr_mat.indptr).type(torch.long)
            self.indices_ = torch.from_numpy(csr_mat.indices).type(torch.long)
        elif indptr is not None and indices is not None:
            if(isinstance(indptr, torch.tensor)):
                self.indptr_ = indptr.type(torch.long)
                self.indices_ = indices.type(torch.long)
            elif(isinstance(indptr, np.ndarray)):
                self.indptr_ = torch.from_numpy(indptr).type(torch.long)
                self.indices_ = torch.from_numpy(indices).type(torch.long)
        self.eid_ = eid
        self.feature_order_ = None

    @property
    def indptr(self):
        return self.indptr_
    
    @property
    def indices(self):
        return  self.indices_
    
    @property
    def eid(self):
        return self.eid_
    
    @property
    def feature_order(self):
        return self.feature_order_
    
    @property
    def degree(self):
        return self.indptr_[1:] - self.indptr_[:-1]
    
    @property
    def node_count(self):
        return self.indptr_.shape[0] - 1
    
    @property
    def edge_count(self):
        return self.indices_.shape[0]
    
    def share_memory_(self):
        """
        Place the CSRtopo in shared memory
        将属性数据放置在共享内存中，以便多个进程可以共享这些数据而不必复制它们。这通常用于提高多进程程序的性能和效率。
        """
        self.indptr_.share_memory_()
        self.indices_.share_memory_()
        if self.eid_ is not None:
            self.eid_.share_memory_()
        if self.feature_order_ is not None:
            self.feature_order_.share_memory_()

    @feature_order.setter
    def feature_order(self, feature_order):
        self.feature_order_ = feature_order

    
def reindex_by_config(adj_csr: CSRTopo, graph_feature, gpu_portion):
    node_count = adj_csr.node_count
    cached_count = int(node_count * gpu_portion)
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(cached_count, dtype=torch.long)    #random permutation
    # sort and shuffle
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    _, prev_order = torch.sort(degree, descending=True)   #A tuple of (sorted_tensor, sorted_indices) is returned, where the sorted_indices are the indices of the elements in the original input tensor
    new_order = torch.zeros_like(prev_order)
    prev_order[:cached_count] = prev_order[perm_range]   # 对prev_order中的前gpu_portion%进行乱序重排
    new_order[prev_order] = total_range                  # new_order意为index为i的feature其对应节点度数排名为new_order[i]
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order

def reindex_feature(graph:CSRTopo, feature, cache_ratio):
    assert isinstance(graph, CSRTopo), "Input graph should be CSRtopo object"
    feature, new_order = reindex_by_config(graph, feature, cache_ratio)
    return feature, new_order

def init_p2p(device_list: List[int]):
    """Try to enable p2p acess between devices in device_list

    Args:
        device_list (List[int]): device list
    """
    propeller.init_p2p(device_list)

UNITS = {
    #
    "KB": 2**10,
    "MB": 2**20,
    "GB": 2**30,
    #
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
}

def parse_size(sz):
    """
    返回输入参数对应的字节数
    """
    if isinstance(sz, int):
        return sz
    elif isinstance(sz, float):
        return int(sz)
    elif isinstance(sz, str):
        for suf, u in sorted(UNITS.items()):
            if sz.upper().endswith(suf):
                return int(float(sz[:-len(suf)]) * u)         # eg: sz = 200M  return 200 * 2**20
    raise Exception("invalid size: {}".format(sz))

