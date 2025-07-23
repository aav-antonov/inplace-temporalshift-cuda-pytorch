#pragma once
#include <torch/extension.h>

void temporal_shift_inplace_forward(at::Tensor& input, int64_t fold) ;
void temporal_shift_inplace_backward(at::Tensor& grad_output, int64_t fold) ;



