"""Setup configuration for Lexora SDK."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lexora",
    version="0.1.1",
    author="VesperAkshay",
    author_email="vesperakshay@gmail.com",
    description="A production-ready, plug-and-play Python SDK for building intelligent RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VesperAkshay/lexora",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0.0",
        "litellm>=1.0.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.0",
        "pinecone-client>=3.0.0",
        "chromadb>=0.4.0",
        "openai>=1.0.0",
        "tiktoken>=0.5.0",
        "asyncio-throttle>=1.0.0",
        "tenacity>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lexora=lexora.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)