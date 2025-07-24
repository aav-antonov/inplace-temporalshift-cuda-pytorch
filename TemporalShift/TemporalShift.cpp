#include <torch/serialize.h>
#include <ATen/core/symbol.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <iostream>
#include "TemporalShift.h"  // include the declarations


// Explicit registration function
TORCH_LIBRARY(tsm2, m) {

    m.def("tsm_inplace (Tensor(a!) input, int fold, int forward) -> ()", &temporal_shift_inplace);

}

// Separate implementation for CUDA
TORCH_LIBRARY_IMPL(tsm2, CUDA, m) {
    m.impl("tsm_inplace", TORCH_FN(temporal_shift_inplace));

}

