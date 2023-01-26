#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlc2nwb",
    version="0.3",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="DeepLabCut <-> NWB conversion utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DLC2NWB",
    install_requires=[
        "ndx-pose>=0.1.1",
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
