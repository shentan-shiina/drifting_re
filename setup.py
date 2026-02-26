from setuptools import setup, find_packages

setup(
    name="drifting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "hydra-core>=1.3",
        "tqdm>=4.66",
    ],
)