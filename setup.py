#!/usr/bin/env python
"""Build the Brain Workshop C extension in-place:

    python setup.py build_ext --inplace
"""
import os
import sys
from setuptools import Extension, setup

here = os.path.dirname(os.path.abspath(__file__))

if os.name == 'nt':
    extra_compile_args = ['/O2']
    extra_link_args = []
else:
    extra_compile_args = ['-O3', '-std=c11']
    extra_link_args = ['-lm']

bwcore = Extension(
    'bwcore',
    sources=[os.path.join('native', 'bwcore.c')],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='neural-workshop-bwcore',
    version='5.0',
    description='C kernels for Neural Workshop hot loops',
    ext_modules=[bwcore],
)
