"""
Setup script for LGD Mapping application.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Separate development requirements
dev_requirements = [req for req in requirements if any(dev in req for dev in ["pytest", "black", "flake8", "mypy", "sphinx"])]
install_requirements = [req for req in requirements if req not in dev_requirements]

setup(
    name="lgd-mapping",
    version="1.0.0",
    author="Data Analytics Team",
    description="A modular application for mapping entities to LGD codes",
    long_description="LGD Mapping Refactor - A well-structured, modular Python application for mapping village, block, and district data to their corresponding LGD codes.",
    packages=find_packages(),
    install_requires=install_requirements,
    extras_require={
        "dev": dev_requirements,
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "lgd-mapping=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)