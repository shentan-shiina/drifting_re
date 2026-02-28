import os
from setuptools import setup, find_packages

def parse_requirements():
    """Read and parse the requirements.txt file dynamically."""
    if not os.path.exists("requirements.txt"):
        return []
        
    with open("requirements.txt", "r") as f:
        lines = f.read().splitlines()
    
    reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Strip local version specifiers (e.g., +cu128) 
        # as they aren't standard PyPI packages and cause setup.py errors
        if "+cu" in line:
            line = line.split("+cu")[0]
        reqs.append(line)
        
    return reqs

setup(
    name="drifting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=parse_requirements(),
)