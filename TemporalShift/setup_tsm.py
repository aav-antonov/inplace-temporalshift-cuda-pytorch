from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tsm2',
    ext_modules=[
        CUDAExtension('tsm_cuda2', [
            'TemporalShift.cpp',
            'TemporalShift_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

#python setup_tsm.py build_ext --inplace --force
