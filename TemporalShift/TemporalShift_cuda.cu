#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "TemporalShift.h"

#include <chrono> // For std::chrono
#include <cuda_runtime.h> // For cudaDeviceSynchronize
#include <cstdio> // For printf

#define threads  1024


template <typename scalar_t>
__device__ void shift_up_fold(
        scalar_t* __restrict__ input,
        const int n,
        const int s,
        const int c,
        const int h,
        const int w,
        const int bid,
        const int fold
){

    int thread_id = threadIdx.x;
    int stride = blockDim.x;

    int vec_elems = fold * h * w ;

    for (int seg_id = 1; seg_id < s; ++seg_id) {
        scalar_t* input_seg_a = input + bid*s*c*h*w + seg_id *c*h*w ;
        scalar_t* input_seg_b = input + bid*s*c*h*w + (seg_id - 1) *c*h*w ;

        for (int j = thread_id; j < vec_elems; j += stride) {
            input_seg_b[j] = input_seg_a[j];
        }
    }

    scalar_t* input_seg_last = input + bid*s*c*h*w + (s - 1)*c*h*w;
    for (int j = thread_id; j < vec_elems; j += stride) {
        input_seg_last[j] = 0;
    }

}


template <typename scalar_t>
__device__ void shift_down_fold(
        scalar_t* __restrict__ input,
        const int n,
        const int s,
        const int c,
        const int h,
        const int w,
        const int bid,
        const int fold
){

    int thread_id = threadIdx.x;
    int stride = blockDim.x;

    int vec_elems = (fold* h * w) ;

    scalar_t* input_seg_bid = input + bid*s*c*h*w;

    for(int seg_id = s-2; seg_id >= 0; seg_id--){

        scalar_t* input_seg_a = input_seg_bid + seg_id*c*h*w + fold*h*w;
        scalar_t* input_seg_b = input_seg_bid + (seg_id+1)*c*h*w + fold*h*w;

        for (int j = thread_id; j < vec_elems; j += stride) {
            input_seg_b[j] = input_seg_a[j];
        }

    }

    scalar_t* input_seg_last = input_seg_bid + 0*c*h*w + fold*h*w;
    for (int j = thread_id; j < vec_elems; j += stride) {
        input_seg_last[j] = 0;
    }
}
//------------------------------------------//




template <typename scalar_t>
__global__ void temporal_shift_kernel(
        scalar_t* __restrict__ input,
        const int n, //batch_size
        const int s, //segmnet_size
        const int c, //channel_size
        const int h, // image height
        const int w, // image width
        const int fold, // number of chanels to shift up, same number shifted down
        const int forward

) {

    //each block shift fold channels in 1 batch
    if (blockIdx.x >= n ) return;

    const int bid = blockIdx.x ;//batch_id


    if(forward == 1){
        shift_up_fold(input, n,s,c,h,w,bid,fold);
        shift_down_fold(input, n,s,c,h,w,bid,fold);
    }else{
        shift_down_fold(input, n,s,c,h,w,bid,fold);
        shift_up_fold(input, n,s,c,h,w,bid,fold);
    }
}

void temporal_shift_inplace(at::Tensor& input, int64_t fold, int64_t forward) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor (n,t,c,h,w)");

    const int n = input.size(0);
    const int t = input.size(1);
    const int c = input.size(2);
    const int h = input.size(3);
    const int w = input.size(4);

    const int blocks = (n);


    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "temporal_shift_backward", ([&] {
        temporal_shift_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), n, t, c, h, w, fold, forward);

        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in temporal_shift_backward_kernel: %s\n", cudaGetErrorString(err));
        }

    }));

}


