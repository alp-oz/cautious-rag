from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cautious-rag",
    version="0.1.0",
    author="Alp Ozturk",
    author_email="your.email@example.com",
    description="A RAG system that knows when to stay quiet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alp-oz/cautious-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "black"],
    },
)