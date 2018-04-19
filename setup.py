#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from cml_data_readers import __version__

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

# TODO: put package requirements here
requirements = []

# TODO: put setup requirements here
setup_requirements = []

setup(
    name='cml_data_readers',
    version=__version__,
    description="Collection of utility classes for reading CML-specific data files",
    long_description=readme + '\n\n' + history,
    author="Penn Computational Memory Lab",
    url='https://github.com/pennmem/cml_data_readers',
    packages=find_packages(include=['cml_data_readers']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='cml_data_readers',
    setup_requires=setup_requirements,
)
