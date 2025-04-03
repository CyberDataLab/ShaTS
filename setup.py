from setuptools import find_packages, setup

setup(
    name="tsg_shap",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "matplotlib",
    ],
    author="Manuel Franco de la Peña",
    author_email="manuel.francop@um.es",
    description="Package TSG-SHAP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelFranco/TSG-SHAP",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
