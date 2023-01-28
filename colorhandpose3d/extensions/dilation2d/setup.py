from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(name='dilation2d',
      ext_modules=[
          CUDAExtension('dilation2d_cuda', [
              'src/cuda/dilation2d_cuda.cpp',
              'src/cuda/dilation.cu'
          ]),
          CppExtension('dilation2d', [
              'src/cpp/dilation2d.cpp'
          ])
      ],
      cmdclass={'build_ext': BuildExtension})
