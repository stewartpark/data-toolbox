import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-toolbox",
    version="0.0.1",
    author="Stewart Park",
    author_email="hello@stewartjpark.com",
    description="A toolbox for various ML tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stewartpark/data-toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
