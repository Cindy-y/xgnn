#include <torch/extension.h>

void register_cuda_feature(pybind11::module &m);

void register_cuda_comm(pybind11::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    register_cuda_feature(m);
    register_cuda_comm(m);
}