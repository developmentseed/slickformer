#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

# with open("README.rst") as readme_file:
#     readme = readme_file.read()

# with open("HISTORY.rst") as history_file:
#     history = history_file.read()

requirements = [
    "Click>=7.0",
    # "Cython", installing this causes timeout errors during build ???
    # "distancerasters", causes fiona build errors on mac m1
    "pycocotools==2.0.4",
    "pycococreatortools @ git+https://github.com/waspinator/pycococreator.git#egg=pycocreatortools",
    "ipywidgets",
]

setup(
    author="Ryan Avery",
    author_email="ryan@developmentseed.org",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Cerulean",
    entry_points={
        "console_scripts": [
            "ceruleanml=ceruleanml.cli:main",
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    # long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="ceruleanml",
    name="ceruleanml",
    packages=find_packages(include=["ceruleanml", "ceruleanml.*"]),
    test_suite="tests",
    url="https://github.com/rbavery/ceruleanml",
    version="0.1.0",
    zip_safe=False,
)
