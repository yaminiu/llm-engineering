"""Setup configuration for Price Intelligence package."""

from setuptools import setup, find_packages

setup(
    name="price-intelligence",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
