from setuptools import setup, find_packages

setup(
    name="tg_shap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
    ],
    author="Manuel Franco de la Peña",
    author_email="manuel.francop@um.es",
    description="Paquete para usar TG-SHAP",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelFranco/TG-SHAP",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
