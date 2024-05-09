import torch
from .utils import parse_size, Topo
from xgnn import propeller 
class Offset:
    def __init__(self, start, end):
        self.start_ = start
        self.end_ = end

    @property
    def start(self):
        return self.start_
    
    @property
    def end(self):
        return self.end_
    
class SharedTensorConfig:
    def __init__(self, device_memory_budget):       # devcie_memory_budget为字典类型，key为device id, values为内存大小
        self.tensor_offset_device = {}
        self.device_memory_budget_ = device_memory_budget
        for device in device_memory_budget:
            self.device_memory_budget_[device] = parse_size(device_memory_budget[device])
    
    @property
    def device_memory_budget(self):
        return self.device_memory_budget_
    
    @property
    def device_list(self):
        return list(self.device_memory_budget.keys())
    
class SharedTensor:
    def __init__(self, current_device: int, shared_tensor_config: SharedTensorConfig):
        self.shared_tensor = propeller.SharedTensor(current_device)
        self.current_device = current_device
        self.shared_tensor_config = shared_tensor_config or SharedTensorConfig({})
        self.topo = None
        self.current_clique = None

        # cpu part
        self.cpu_tensor = None

    def init_topo(self):
        if self.current_clique is not None:
            return

        device_list = set(self.shared_tensor_config.device_list)
        device_list.add(self.current_device)
        device_list = list(device_list)
        self.topo = Topo(device_list)
        self.current_clique = self.topo.get_clique_id(self.current_device)
    
    def append(self, cpu_tensor, device):

        if device == -1:
            if self.cpu_tensor is not None:
                raise Exception("cpu tensor has been already appended")
            self.cpu_tensor = cpu_tensor
            self.shared_tensor.append(cpu_tensor, -1)
            return
        if self.shared_tensor_config.device_memory_budget.get(device,
                                                             None) is None:
            self.shared_tensor_config.tensor_offset_device[device] = Offset(
                self.shared_tensor.size(0),
                self.shared_tensor.size(0) + cpu_tensor.shape[0])
            self.shared_tensor_config.device_memory_budget[
                device] = cpu_tensor.numel() * cpu_tensor.element_size()
            print(
                f"LOG >>> Memory Budge On {device} is {self.shared_tensor_config.device_memory_budget[device] // 1024 // 1024} MB"
            )
            self.shared_tensor.append(cpu_tensor, device)
        else:
            raise Exception(f"{device} tensor has been already appended")

    def partition(self, tensor, memory_budget):
        """
        Args:
            tensor: pytorch cpu tensor
            memory_budget: memory size in bytes
            
        """
        # 暂时先假设为float tensor
        element_size = tensor.shape[1] * tensor.element_size()
        return memory_budget // element_size

    def from_cpu_tensor(self, tensor):
        cur_pos = 0
        size = 0
        # We Assume Only 2 Numa Node
        for device_id, memory_budget in self.shared_tensor_config.device_memory_budget.items(
        ):
            if cur_pos > tensor.shape[0]:
                break

            size = self.partition(tensor, memory_budget)
            size = min(size, tensor.shape[0] - cur_pos)
            self.shared_tensor.append(tensor[cur_pos:cur_pos + size], device_id)
            device_offset = Offset(cur_pos, cur_pos + size)
            self.shared_tensor_config.tensor_offset_device[
                device_id] = device_offset

            cur_pos += size
            print(
                f"LOG >>> Assign {int(100 * size * 1.0 / tensor.shape[0])}% data to {device_id}"
            )

        if cur_pos < tensor.shape[0]:
            # allocate the rest of data on CPU
            self.cpu_tensor = tensor[cur_pos:]
            self.shared_tensor.append(self.cpu_tensor, -1)
            print(
                f"LOG >>> Assign {100 - int(100 * cur_pos * 1.0 / tensor.shape[0])}% data to CPU"
            )
            del tensor

    def collect_device(self, input_orders, nodes, inter_device, wait_results):

        request_nodes_mask = (
            nodes >=
            self.shared_tensor_config.tensor_offset_device[inter_device].start
        ) & (nodes <
             self.shared_tensor_config.tensor_offset_device[inter_device].end)
        request_nodes = torch.masked_select(nodes, request_nodes_mask)
        part_orders = torch.masked_select(input_orders, request_nodes_mask)
        request_nodes = request_nodes.to(inter_device)

        with torch.cuda.device(inter_device):
            result = self.shared_tensor[request_nodes]
        result = result.to(self.current_device)
        wait_results.append((part_orders, result))

    def __getitem__(self, nodes):

        self.init_topo()
        nodes = nodes.to(self.current_device)

        feature = self.shared_tensor[nodes]

        input_orders = torch.arange(nodes.size(0),
                                    dtype=torch.long,
                                    device=self.current_device)

        # call inter request, we unfold for loop
        inter_clique_devices = self.topo.p2pClique2Device.get(
            1 - self.current_clique, [])

        wait_results = []

        for inter_device in inter_clique_devices:
            if self.shared_tensor_config.tensor_offset_device.get(
                    inter_device, None) is not None:
                self.collect_device(input_orders, nodes, inter_device,
                                    wait_results)

        for result in wait_results:
            feature[result[0]] = result[1]

        return feature

    @property
    def shape(self):
        return self.shared_tensor.shape()

    @property
    def device(self):
        return self.current_device

    def share_ipc(self):
        items = self.shared_tensor.share_ipc()
        gpu_part_ipc_list = [item.share_ipc() for item in items]

        return gpu_part_ipc_list, self.cpu_tensor, self.shared_tensor_config

    def from_ipc_handle(self, gpu_ipc_list, cpu_tensor):
        for gpu_ipc in gpu_ipc_list:
            gpu_item = propeller.ShardTensorItem()
            gpu_item.from_ipc(gpu_ipc)
            self.shared_tensor.append(gpu_item)
        if cpu_tensor is not None:
            self.cpu_tensor = cpu_tensor
            self.shared_tensor.append(cpu_tensor, -1)

    @classmethod
    def new_from_share_ipc(cls, ipc_handles, current_device):
        gpu_part_ipc_list, cpu_tensor, shared_tensor_config = ipc_handles
        shared_tensor = cls(current_device, shared_tensor_config)
        shared_tensor.from_ipc_handle(gpu_part_ipc_list, cpu_tensor)
        return shared_tensor

    def size(self, dim):
        return self.shared_tensor.size(dim)



