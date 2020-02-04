import os
from setuptools import find_packages, setup

_pkg: str = "python_project_skeleton"


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def read_version():
    return read("VERSION").strip()


# Declare minimal set for installation
required_packages = ["boto3", "paramiko>= 2.7.0"]

setup(
    name=_pkg,
    packages=find_packages(),
    entry_points={"console_scripts": [f"{_pkg} = {_pkg}.__main__:main"]},
    version=read_version(),
    description="This package does x,y,z.",
    long_description=read("README.md"),
    author="Firstname Lastname",
    author_email="first.last@email.com",
    url=f"https://github.com/abcd/{_pkg}/",
    download_url="",
    project_urls={
        "Bug Tracker": f"https://github.com/abcd/{_pkg}/issues/",
        "Documentation": f"https://{_pkg}.readthedocs.io/en/stable/",
        "Source Code": f"https://github.com/abcd/{_pkg}/",
    },
    license="MIT",
    keywords="word1 word2 word3",
    platforms=["any"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6.0",
    install_requires=required_packages,
)
