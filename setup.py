#!/usr/bin/env python
from setuptools import setup
import sys

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(name='DLITE',
      description=' a dynamic force-inference model to estimate tensions in colony time-series',
      author='Vasan, Ritvik and Maleckar, Mary M. and Williams, C. David. and Rangamani, Padmini',
      author_email='cdave@alleninstitute.org',
      url='https://github.com/AllenCellModeling/DLITE',
      packages=['cell_soap'],
      license='LICENSE.txt',
      version='0.1.0',
      keywords=['Force-inference', 'tension', 'dynamic' ],
      install_requires=required)
