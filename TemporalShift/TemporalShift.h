#pragma once
#include <torch/extension.h>

void temporal_shift_inplace(at::Tensor& input, int64_t fold, int64_t forward) ;
//void temporal_shift_inplace_backward(at::Tensor& grad_output, int64_t fold) ;

//void temporal_shift_inplace_forward_vect(at::Tensor& input, int64_t fold) ;
//void temporal_shift_inplace_backward_vect(at::Tensor& grad_output, int64_t fold) ;



