"""Placeholder."""
import os
from typing import List

from setuptools import find_packages, setup

import versioneer

_pkg: str = "smepu"


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Declare minimal set for installation
required_packages: List[str] = []

# Specific use case dependencies
extras = {
    "all": ["click"],
    "click": ["click"],
}

setup(
    name=_pkg,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Utilities for Amazon SageMaker's training entrypoint script.",
    long_description=read("README.md"),
    author="Verdi March",
    # author_email="first.last@email.com",
    url=f"https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities/",
    download_url="",
    project_urls={
        "Bug Tracker": f"https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities/issues/",
        "Source Code": f"https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities/",
    },
    license="MIT",
    keywords="Amazon SageMaker train entrypoint",
    platforms=["any"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6.0",
    install_requires=required_packages,
    extras_require=extras,
    include_package_data=True,
)
