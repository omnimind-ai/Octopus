from setuptools import setup
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='octopus_io',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='octopus_io',  # Python module name after build
            sources=[str(Path(__file__).with_name('io_backend.cpp').resolve())],
            extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': ['-O3']},
            extra_link_args=['-luring']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    py_modules=[],
)
