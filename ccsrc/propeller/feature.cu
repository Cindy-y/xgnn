#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>

#include "include/propeller.cu.hpp"
#include "include/shared_tensor.cu.hpp"
#include "include/common.hpp"


#include <torch/extension.h>
#include <atomic>
#include <iostream>
#include <string>
#include <torch/csrc/utils/python_numbers.h>
#include <unordered_map>

namespace propeller{
#define CHECK_CPU(x) AT_ASSERTM(!x.device().is_cuda(), #x"must be CPU tensor") //#x 将参数 x 转换为字符串字面值，以便将变量名包含在错误消息中。
class SharedTensorItem{
public:
    int device;
    cudaIpcMemHandle_t mem_handle;
    std::vector<int> shape;
    //for now we assume it is all float
    int element_size;

    SharedTensorItem(int device_, cudaIpcMemHandle_t mem_handle_, std::vector<int> shape_)
        : device(device_), mem_handle(mem_handle_), shape(shape_) {}

    SharedTensorItem(){}

    std::tuple<int, int, py::bytes, std::vector<int>> share_ipc(){
        auto _handle = PyBytes_FromStringAndSize((char*)&mem_handle, 
                                                CUDA_IPC_HANDLE_SIZE);  //将指向 mem_handle 的地址的字符型指针转换为 Python 字节对象
        auto bytes_obj = py::reinterpret_steal<py::object>((PyObject *)_handle);  //将Python字节对象的指针 _handle 转换为 pybind11 中的 py::object 对象(c++对象)
        return std::make_tuple(device, element_size, bytes_obj, shape);
    }

    void from_ipc(std::tuple<int, int, std::string, std::vector<int>> ipc_data){
        device = std::get<0>(ipc_data);
        element_size = std::get<1>(ipc_data);
        shape = std::get<3>(ipc_data);
        auto handle = std::get<2>(ipc_data);
        auto ipc_handle =
            reinterpret_cast<const cudaIpcMemHandle_t *>(handle.c_str());    //c_str() 返回的是指向字符数组的指针。

        mem_handle = *ipc_handle;
    }
};

class SharedTensor{
private:
    int device_;
    bool inited_;
    int device_count_;
    int element_size;
    int numa_broker_device;
    std::vector<int64_t> shape_;
    std::vector<int64_t> offset_list_;
    std::vector<void *> dev_ptrs_;
    std::vector<int> tensor_devices_;
    std::vector<int> access_book;
    std::vector<std::vector<int>> tensor_shapes_;
    std::unordered_map<int, std::tuple<char **, int64_t *, int *>> device_pointers_map;
public:
    SharedTensor(int device) : device_(device), inited_(false), device_count_(0) 
    {
        offset_list_.push_back(0);
    }
    size_t get_tensor_bytes(torch::Tensor tensor){
        //assume it's float    eg:tensor([[1,2,3], [4,5,6]],[[1,2,3], [4,5,6]])
        int dim = tensor.dim();    //dim = 3表示有3个维度   tensor.size [2,3,3] 
        size_t total_bytes = element_size;   //size_t 是一种用于表示对象的大小或长度的数据类型。它是一个无符号整数类型，通常被用于处理内存分配、数组索引等与数据存储空间大小相关的操作。
        for(int index = 0; index < dim; index++){
            total_bytes *= tensor.sizes()[index];
        }
        return total_bytes;
    }
    std::vector<int> get_tensor_shape(torch::Tensor tensor){
        std::vector<int> shape;
        int dim = tensor.dim();
        for(int index = 0; index < dim; index++){
            shape.emplace_back(tensor.sizes()[index]);
        }
        return shape;
    }
    void append(SharedTensorItem item){
        cudaSetDevice(device_);
        if(!inited_){
            shape_.resize(item.shape.size());   //size()取元素个数值
            shape_[0] = 0;
            auto tensor_sizes = item.shape;
            for(int index = 1; index < shape_.size(); index++){
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        offset_list_.emplace_back(offset_list_[offset_list_.size()-1] + item.shape[0]);   //offset_list_[offset_list_.size()-1]表示数组最后一个元素值
        // check assessbility
        if(item.device >=0){
            int access_i_j, access_j_i;
            cudaDeviceCanAccessPeer(&access_i_j, device_, item.device);
            cudaDeviceCanAccessPeer(&access_j_i, item.device, device_);
            if((access_i_j && access_j_i) || item.device == device_){
                access_book.push_back(1);
            }
            else{
                access_book.push_back(0);
            }
        }          //item在GPU
        else{
            access_book.push_back(1);
        }          //item在CPU

        // get dev_ptr that can be accessed from this process
        void* ptr = NULL;
        tensor_devices_.push_back(item.device);
        if(!access_book[access_book.size()-1]){
            cudaSetDevice(item.device);
            cudaIpcOpenMemHandle(&ptr, item.mem_handle, cudaIpcMemLazyEnablePeerAccess);
            cudaSetDevice(device_);
        }
        else{
            cudaIpcOpenMemHandle(&ptr, item.mem_handle, cudaIpcMemLazyEnablePeerAccess);
        }
        dev_ptrs_.push_back(ptr);
        element_size = item.element_size;
        shape_[0] += item.shape[0];
        device_count_ +=1;
        cudaCheckError();
    }

    void append(torch::Tensor &tensor, int target_device)
    {
        CHECK_CPU(tensor);
        // for now, we assume tensor is added ordered
        if (!inited_) {
            shape_.resize(tensor.dim());
            shape_[0] = 0;
            auto tensor_sizes = tensor.sizes();
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        element_size = tensor.element_size();
        tensor_shapes_.push_back(get_tensor_shape(tensor));

        offset_list_.push_back(offset_list_[offset_list_.size() - 1] +
                               tensor.sizes()[0]);

        void *ptr = NULL;
        size_t data_size = get_tensor_bytes(tensor);
        tensor_devices_.push_back(target_device);
        if (target_device >= 0) {
            // if target_device >= 0, it means we use p2p
            // printf("LOG >>> Malloc Data On Device %d With %ulld Bytes\n",
            // target_device, data_size);
            cudaSetDevice(target_device);
            cudaMalloc(&ptr, data_size);
            cudaMemcpy(ptr, tensor.data_ptr(), data_size,
                       cudaMemcpyHostToDevice);
            cudaSetDevice(device_);

            // decide access book

            int access_i_j, access_j_i;
            cudaDeviceCanAccessPeer(&access_i_j, device_, target_device);
            cudaDeviceCanAccessPeer(&access_j_i, target_device, device_);
            if ((access_i_j && access_j_i) || device_ == target_device) {
                access_book.push_back(1);
                // printf("%d <-> %d support peer access \n", device_,
                // target_device);
            } else {
                access_book.push_back(0);
                // printf("%d <-> %d dont support peer access \n", device_,
                // target_device);
            }

        } else {
            cudaSetDevice(device_);
            // if target_device < 0, it means we use Zero-Copy

            propellerRegister(tensor.data_ptr(), data_size,
                           cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&ptr, (void *)tensor.data_ptr(), 0);

            // cudaMemcpy(&ptr, (void*)tensor.data_ptr(), data_size, cudaMemcpyHostToDevice);
            //ptr =  (void*)tensor.data_ptr();

            access_book.push_back(1);
            // printf("%d <-> CPU support peer access \n", device_);

        }

        dev_ptrs_.push_back(ptr);

        shape_[0] += tensor.size(0);
        device_count_ += 1;
    }
    
    std::tuple<char**, int64_t*, int *> get_device_pointers(int device){
        auto iter  = device_pointers_map.find(device);
        if(iter == device_pointers_map.end()){
            char** buffers_device;
            int64_t* offset_device;
            int* access_book_device;

            //copy buffers Device
            cudaMalloc((void***)&buffers_device, sizeof(float*) * device_count_);
            cudaMemcpy(buffers_device, &dev_ptrs_[0], sizeof(float*) * dev_ptrs_.size(), cudaMemcpyHostToDevice);
            cudaCheckError();

            //copy offset 
            cudaMalloc((void**)&offset_device, sizeof(int64_t*) * offset_list_.size());
            cudaMemcpy(offset_device, &offset_list_[0], sizeof(int64_t*) * offset_list_.size(), cudaMemcpyHostToDevice);
            cudaCheckError();

            //copy accessbook
            cudaMalloc((void **)&access_book_device, sizeof(int) * access_book.size());
            cudaMemcpy(access_book_device, &access_book[0], sizeof(int) * access_book.size(), cudaMemcpyHostToDevice);
            cudaCheckError();

            device_pointers_map.emplace(device, std::make_tuple(buffers_device, offset_device, access_book_device));
            iter = device_pointers_map.find(device);
        }
        return iter->second;
    }

    torch::Tensor operator[](torch::Tensor &indices){
        int current_device = 0;
        cudaGetDevice(&current_device);
        auto stream = at::cuda::getCurrentCUDAStream();

        std::vector<int64_t> res_shape(shape_);
        res_shape[0] = indices.numel();

        auto options = torch::TensorOptions();
        if(element_size == 2){
            options = options.dtype(torch::kFloat16).device(torch::kCUDA, current_device);
        }else if(element_size == 4){
            options = options.dtype(torch::kFloat32).device(torch::kCUDA, current_device);
        }

        auto res = torch::empty(res_shape, options);
        cudaCheckError();

        char** buffers_device;
        int64_t* offset_device;
        int* access_book_device;

        auto val = get_device_pointers(current_device);
        buffers_device = std::get<0>(val);
        offset_device = std::get<1>(val);
        access_book_device = std::get<2>(val);

        int blockSize = 0;
        int numBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, propeller_tensor_gather);

        int ignore_access_book = 0;
        if(current_device != device_){
            ignore_access_book = 1;
        }
        propeller_tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
            buffers_device, offset_device, offset_list_.size(),
            indices.data_ptr<int64_t>(), indices.numel(), (char*)res.data_ptr(),
            stride_in_bytes(0), access_book_device, ignore_access_book);
        
        cudaCheckError();
        return res;
    }

    std::vector<int64_t> shape() const { return shape_; }

    int device() const { return device_; }

    int size(int dim) const
    {
        if (shape_.size() == 0) return 0;
        return shape_[dim];
    }

    int64_t stride(int dim) const
    {
        int64_t res = 1;
        for (int index = dim + 1; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }

    int64_t stride_in_bytes(int dim) const{
        return stride(dim) * element_size;
    }

    int64_t numel() const
    {
        int64_t res = 1;
        for (int index = 0; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }
    std::vector<SharedTensorItem> share_ipc()
    {
        std::vector<SharedTensorItem> res;
        for (int index = 0; index < dev_ptrs_.size(); index++) {
            if (tensor_devices_[index] >= 0) {
                cudaSetDevice(tensor_devices_[index]);
                SharedTensorItem *item = new SharedTensorItem();
                item->device = tensor_devices_[index];
                item->shape = tensor_shapes_[index];
                item->element_size = element_size;
                cudaIpcGetMemHandle(&(item->mem_handle), dev_ptrs_[index]);
                res.push_back(*item);
            }
        }
        return res;
    }

    int device_count() const { return device_count_; }

    void unregister(torch::Tensor &cpu_tensor)
    {

        std::cout << "begin unregister" << std::endl;
        cudaHostUnregister((void *)cpu_tensor.data_ptr<float>());
        std::cout << "end unregister" << std::endl;
    }
};

void init_p2p(std::vector<int> devices)
{
    std::cout << "LOG>>> P2P Access Initilization" << std::endl;

    for (int i = 0; i < devices.size(); i++) {
        int src = devices[i];
        cudaSetDevice(src);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, src);

        // CUDA IPC is only supported on devices with unified addressing
        if (!prop.unifiedAddressing) {
            printf(
                "Device %d does not support unified addressing, skipping...\n",
                i);
            continue;
        }
        // This sample requires two processes accessing each device, so we need
        // to ensure exclusive or prohibited mode is not set
        if (prop.computeMode != cudaComputeModeDefault) {
            printf(
                "Device %d is in an unsupported compute mode for this sample\n",
                i);
            continue;
        }

        for (int j = i + 1; j < devices.size(); j++) {
            int dst = devices[j];
            int access_i_j = 0;
            int access_j_i = 0;
            cudaDeviceCanAccessPeer(&access_i_j, src, dst);
            cudaDeviceCanAccessPeer(&access_j_i, dst, src);
            if (access_i_j && access_j_i) {
                printf("Enable P2P Access Between %d <---> %d \n", src, dst);
                cudaSetDevice(src);
                cudaDeviceEnablePeerAccess(dst, 0);
                cudaCheckError();
                cudaSetDevice(dst);
                cudaDeviceEnablePeerAccess(src, 0);
                cudaCheckError();
            }
        }
    }
}

bool can_device_access_peer(int src_device_index, int dst_device_index)
{
    int access_i_j = 0, access_j_i = 0;
    cudaDeviceCanAccessPeer(&access_i_j, src_device_index, dst_device_index);
    cudaDeviceCanAccessPeer(&access_j_i, dst_device_index, src_device_index);
    return (access_i_j == 1) && (access_j_i == 1);
}

}

void register_cuda_feature(pybind11::module &m){

    m.def("init_p2p", &propeller::init_p2p, py::call_guard<py::gil_scoped_release>());

    m.def("can_device_access_peer", &propeller::can_device_access_peer, py::call_guard<py::gil_scoped_release>());

    py::class_<propeller::SharedTensorItem>(m, "SharedTensorItem")
        .def(py::init<>())
        .def("share_ipc", &propeller::SharedTensorItem::share_ipc)
        .def("from_ipc", &propeller::SharedTensorItem::from_ipc);
    
    py::class_<propeller::SharedTensor>(m, "SharedTensor")
        .def(py::init<int>())
        .def("__getitem__", &propeller::SharedTensor::operator[],
             py::call_guard<py::gil_scoped_release>())
        .def("unregister", &propeller::SharedTensor::unregister,
             py::call_guard<py::gil_scoped_release>())
        .def("shape", &propeller::SharedTensor::shape,
             py::call_guard<py::gil_scoped_release>())
        .def("numel", &propeller::SharedTensor::numel,
             py::call_guard<py::gil_scoped_release>())
        .def("device", &propeller::SharedTensor::device,
             py::call_guard<py::gil_scoped_release>())
        .def("stride", &propeller::SharedTensor::stride,
             py::call_guard<py::gil_scoped_release>())
        .def("size", &propeller::SharedTensor::size,
             py::call_guard<py::gil_scoped_release>())
        .def("device_count", &propeller::SharedTensor::device_count,
             py::call_guard<py::gil_scoped_release>())
        .def("append",
             py::overload_cast<torch::Tensor &, int>(
                 &propeller::SharedTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("append",
             py::overload_cast<propeller::SharedTensorItem>(
                 &propeller::SharedTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("share_ipc", &propeller::SharedTensor::share_ipc,
             py::call_guard<py::gil_scoped_release>());
}


