#!/user/bin/env

# author : caleb7023

from setuptools import setup, find_packages

setup(
    name="nn_calc_lib",
    version="1.0.0",
    packages=find_packages(),
    author="caleb7023",
    description="A lib for neural network calculation",
    install_requires=[
        "cupy",
    ],
)