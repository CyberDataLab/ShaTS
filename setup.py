from setuptools import setup, find_packages

setup(
    name="ts_shap",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
    ],
    author="Manuel Franco de la Peña",
    author_email="manuel.francop@um.es",
    description="Paquete para usar TS-SHAP",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelFranco/TS-SHAP",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
