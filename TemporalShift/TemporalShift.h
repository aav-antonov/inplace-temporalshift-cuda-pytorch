#pragma once
#include <torch/extension.h>

void temporal_shift_inplace(at::Tensor& input, int64_t fold, int64_t forward, int64_t vect) ;



