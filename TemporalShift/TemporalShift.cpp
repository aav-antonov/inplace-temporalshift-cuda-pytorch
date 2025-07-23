#include <torch/serialize.h>
#include <ATen/core/symbol.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <iostream>
#include "TemporalShift.h"  // include the declarations


// Explicit registration function
TORCH_LIBRARY(tsm2, m) {

    m.def("forward_ts(Tensor(a!) input, int fold) -> ()", &temporal_shift_inplace_forward);
    m.def("backward_ts(Tensor(a!) input, int fold) -> ()", &temporal_shift_inplace_backward);

    //m.def("forward_ts(Tensor(a!) input, int fold) -> ()", &temporal_shift_inplace_forward_vect);
    //m.def("backward_ts(Tensor(a!) input, int fold) -> ()", &temporal_shift_inplace_backward_vect);
}

// Separate implementation for CUDA
TORCH_LIBRARY_IMPL(tsm2, CUDA, m) {
    m.impl("forward_ts", TORCH_FN(temporal_shift_inplace_forward));
    m.impl("backward_ts", TORCH_FN(temporal_shift_inplace_backward));

    //m.impl("forward_ts_vect", TORCH_FN(temporal_shift_inplace_forward_vect));
    //m.impl("backward_ts_vect", TORCH_FN(temporal_shift_inplace_backward_vect));
}

