"""
ClariAI Setup Script

Installation script for the ClariAI audio quality analysis platform.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clariai",
    version="1.0.0",
    author="ClariAI Team",
    author_email="team@clariai.com",
    description="Professional Audio Quality Analysis Platform using LangChain and Machine Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kernelseed/audio-call-quality-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "clariai-analyze=audio_call_quality_model:main",
            "clariai-train=training_pipeline:main",
            "clariai-upload=huggingface_integration:main",
            "clariai-config=config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "clariai": ["*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "audio",
        "quality-analysis",
        "machine-learning",
        "langchain",
        "huggingface",
        "speech-processing",
        "call-quality",
        "ai",
        "ml",
    ],
    project_urls={
        "Bug Reports": "https://github.com/kernelseed/audio-call-quality-analyzer/issues",
        "Source": "https://github.com/kernelseed/audio-call-quality-analyzer",
        "Documentation": "https://github.com/kernelseed/audio-call-quality-analyzer#readme",
        "Changelog": "https://github.com/kernelseed/audio-call-quality-analyzer/blob/main/CHANGELOG.md",
    },
)