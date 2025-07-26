# In-Place TemporalShift Operator: A PyTorch CUDA Custom Kernel Implementation

This repository contains an optimized in-place implementation of the 
TemporalShift operation for video understanding models, 
implemented as a custom CUDA kernel operator for PyTorch. 
Based on the original Temporal Shift Module ([MIT-Han-Lab/TSM](https://github.com/mit-han-lab/temporal-shift-module)) 
and Temporal Shift with Audio Modality ([TSAM](https://github.com/aav-antonov/TSAM)) 


## Key Features

- ✅ **In-place operation** reducing memory usage by 50%
- ⚡ **2-3x faster** than out-of-place implementation
- 🧪 **Rigorous testing** with random parameter validation
- 📊 **Comprehensive benchmarks** for memory and performance

## Installation

```bash
git clone https://github.com/aav-antonov/inplace-temporalshift-cuda-pytorch.git
cd inplace-temporalshift-cuda-pytorch/TemporalShift
python setup_tsm.py build_ext --inplace --force
```

## Benchmark Results

### Correctness Verification
- 10/10 test cases passed with random configurations
- Validated across diverse tensor shapes (batch sizes 1-20, channels 8-128, spatial dims 16-64)
- All comparisons passed with tolerance of 1e-6

### Memory Efficiency (100 batch size, 16 segments, 64 channels, 32x32 spatial)
| Mode          | Peak Memory (MB) | Memory Reduction |
|---------------|------------------|------------------|
| Out-of-place  | 800.0            | 1.00x (baseline) |
| In-place      | 400.0            | 2.00x less       |



### Expected Performance Speedup In-place vs Out-of-place

**Configuration**:
- Batch size: 10  
- Temporal segments: 16  
- Channels: 256  
- Spatial dimensions: 32×32  
- Fold division: 4  
- Benchmark runs: 100  

| Mode                     | Forward Speedup | Backward Speedup |
|--------------------------|----------------:|-----------------:|
| In-place (non-vectored)  | 1.80×           | 1.23×            |
| In-place (vectored)      | 2.85×           | 1.17×            |

## Usage

```python
from TemporalShift.TemporalShift import TemporalShift

# Input tensor (B*T, C, H, W)
x = torch.randn(100*16, 64, 32, 32).cuda()
x.requires_grad_()

# In-place operation (recommended)
x_shifted = TemporalShift.shift(x, n_segment=16, fold_div=4, inplace=True)

# Out-of-place operation
x_shifted = TemporalShift.shift(x, n_segment=16, fold_div=4, inplace=False)
```

### Requirements

- **PyTorch**: 1.7 or higher
- **GPU**: CUDA 10.2+ compatible
- **Python**: 3.6 or higher
- **NVIDIA CUDA Toolkit**: Required (version should match PyTorch CUDA version)