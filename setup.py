#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

test_requirements = [
    'codecov',
    'flake8',
    'pytest',
    'pytest-cov',
    'pytest-raises',
]

setup_requirements = ['pytest-runner']

dev_requirements = [
    'bumpversion>=0.5.3',
    'wheel>=0.33.1',
    'flake8>=3.7.7',
    'tox>=3.5.2',
    'coverage>=5.0a4',
    'Sphinx>=2.0.0b1',
    'sphinx_rtd_theme',
    'twine>=1.13.0',
    'pytest>=4.3.0',
    'pytest-cov==2.6.1',
    'pytest-raises>=0.10',
    'pytest-runner>=4.4',
]

interactive_requirements = [
    'altair',
    'jupyterlab',
    'matplotlib',
    'networkx',
]

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

extra_requirements = {
    'test': test_requirements,
    'setup': setup_requirements,
    'dev': dev_requirements,
    'interactive': interactive_requirements,
    'all': [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements
    ]
}

setup(
    author="Ritvik Vasan",
    author_email='rvasan@eng.ucsd.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Allen Institute Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A dynamic force-inference model to estimate tensions in colony time-series",
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='DLITE',
    name='DLITE',
    packages=find_packages(),
    python_requires=">=3.6",
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url='https://github.com/AllenCellModeling/DLITE',
    version='0.1.0',
    zip_safe=False,
)
