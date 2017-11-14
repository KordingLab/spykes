#! /usr/bin/env python
from setuptools import find_packages
from setuptools import setup

DISTNAME = 'spykes'
DESCRIPTION = """Basic tools for neural data analysis and visualization."""
MAINTAINER = 'Pavan Ramkumar and Hugo Fernandes'
MAINTAINER_EMAIL = 'pavan.ramkumar@gmail.com'
LICENSE = 'MIT'
URL = 'https://github.com/KordingLab/spykes.git'
VERSION = '0.3.dev'

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        long_description=open('README.md').read(),
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'pandas',
            'requests',
        ],
        extras_require={
            'deepdish': ['deepdish'],
            'develop': [
                'nose',
                'coverage',
                'flake8',
                'sphinx',
                'sphinx-gallery',
                'sphinx_rtd_theme',
                'm2r',
                'deepdish',
                'image',
                'tensorflow>=1.4.0',
                'h5py',
            ],
            'ml': [
                'tensorflow>=1.4.0',
            ],
        },
        platforms='any',
        packages=find_packages(exclude=['tests', 'tests.*']),
    )
