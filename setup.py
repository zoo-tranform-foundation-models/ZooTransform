from setuptools import setup, find_packages

setup(
    name="zoo_transform",
    version="0.1.0",
    description="Foundation model for species-specific biological data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "requests>=2.20.0",
        "numpy",
        "pandas==1.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)