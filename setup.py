"""Setup configuration for JetsonZoom package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = (
    [line.strip() for line in requirements_file.read_text().splitlines()
     if line.strip() and not line.startswith("#")]
    if requirements_file.exists()
    else []
)

setup(
    name="jetson-zoom",
    version="1.0.0",
    author="JetsonZoom Team",
    description="Realtime camera control application for Jetson Orin NX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/JetsonZoom",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jetson-zoom=jetson_zoom.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: System :: Hardware",
    ],
)
