import sys
import os
import pathlib
from setuptools import setup, find_packages

root_dir = os.path.dirname(os.path.realpath(__file__))

HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = ["torch", "omegaconf", "hydra-core", "gym"]

# Installation operation
setup(
    name="PWM",
    author="anonymized",
    author_email="anonymized",
    version="0.1.0",
    long_description=README,
    long_description_content_type="text/markdown",
    url="anonymized",
    description="A library for for First-order Reinforcement Learning algorithms",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    package_dir={"": "src"},
    packages=find_packages(
        where="src", exclude=["*.tests", "*.tests.*", "tests.*", "tests", "externals"]
    ),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)

# EOF
