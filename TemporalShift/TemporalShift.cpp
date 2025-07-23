#include <torch/serialize.h>
#include <ATen/core/symbol.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <iostream>
#include "TemporalShift.h"  // include the declarations


// Explicit registration function
TORCH_LIBRARY(tsm2, m) {

    //std::cout << "REGISTERING OPERATORS" << std::endl;  // Debug print
    m.def("forward_ts(Tensor(a!) input, int fold) -> ()",
        &temporal_shift_inplace_forward);

    //std::cout << "Registering backward_ts" << std::endl;
    m.def("backward_ts(Tensor(a!) input, int fold) -> ()",
        &temporal_shift_inplace_backward);
    //std::cout << "REGISTERING OPERATORS COMPLETE" << std::endl;  // Debug print
}

// Separate implementation for CUDA
TORCH_LIBRARY_IMPL(tsm2, CUDA, m) {
    m.impl("forward_ts", TORCH_FN(temporal_shift_inplace_forward));
    m.impl("backward_ts", TORCH_FN(temporal_shift_inplace_backward));
}

