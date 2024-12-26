from setuptools import setup, find_packages

setup(
    name="Graphion",
    version="0.1",
    description="A library for Graph Neural Networks (GNNs) implemented from scratch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repository/Graphion",
    packages=find_packages(exclude=["examples", "tests"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
