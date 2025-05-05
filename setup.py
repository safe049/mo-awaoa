from setuptools import setup, find_packages

setup(
    name="mo-awaoa",
    version="1.2.5",
    author="safe049",
    author_email="safe049@163.com",
    description="Multi-Objective Adaptive Whale Optimization Algorithm (MO-AWAOA)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/safe049/mo-awaoa",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "tqdm>=4.60"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)