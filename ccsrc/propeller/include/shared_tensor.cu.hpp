#pragma once
#include <stdio.h>
#include <torch/extension.h>

#define WARP_SIZE 32

__device__ int find(const int64_t *offsets, const int device_count,
                    const int64_t index)
{
    if(index < 0){
    	return -1;
    }
    int i = 1;
    for (i = 1; i <= device_count; i++) {
        if (index < offsets[i]) { return i - 1; }
    }
    return -1;
}
__global__ void propeller_tensor_gather(char **dev_ptrs, const int64_t *offsets,
                                     const int device_count,
                                     const int64_t *indices, int indice_length,
                                     char *res, const int stride,
                                     const int *access_book,
                                     const int ignore_access_book)
{

    //
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / WARP_SIZE;
    unsigned int warp_step = step / WARP_SIZE;

    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % WARP_SIZE;    //表示当前线程所属线程组中的起始id

    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    char *dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;

    unsigned int local_start = thread_start;          
    while (warp_start < indice_length) {                // each warp take charge of one-feature copy
        local_start = thread_start;
        dev_index = find(offsets, device_count, indices[warp_start]);
        // we only copy data from reachable device
        if (dev_index != -1 && (ignore_access_book || access_book[dev_index] == 1)) {
            dev_ptr = dev_ptrs[dev_index];
            dev_offset = indices[warp_start] - offsets[dev_index];
            src_copy_start = dev_offset * stride;
            dst_copy_start = warp_start * stride;
            for (; local_start < stride; local_start += WARP_SIZE) {
                res[dst_copy_start + local_start] =
                    dev_ptr[src_copy_start + local_start];
            }
        }
        warp_start += warp_step;
    }
}