from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "kmeans_cpp",
        ["app/kmeans.cpp"],
        extra_compile_args=["-std=c++20"]
    ),
]

setup(
    name="kmeans_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
