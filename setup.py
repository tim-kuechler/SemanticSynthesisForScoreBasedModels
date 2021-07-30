from setuptools import setup, find_packages
from torch.utils import cpp_extension
import os


module_path = os.path.dirname(__file__)
setup(
    name='score-sde',
    version='0.0.1',
    description='Semantic Synthesis with Score-Based Generative Models',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'ninja',
        'ml_collections',
        'albumentations',
        'opencv-python'
    ],
    ext_modules=[cpp_extension.CUDAExtension(name="upfirdn2d",
                            sources=["op" + os.sep + "upfirdn2d.cpp", "op" + os.sep + "upfirdn2d_kernel.cu"], include_dirs=cpp_extension.include_paths(),)],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
