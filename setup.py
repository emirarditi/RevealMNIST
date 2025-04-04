from setuptools import setup, find_packages

setup(
    name="reveal_mnist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
    ]
)