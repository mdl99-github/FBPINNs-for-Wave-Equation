from setuptools import setup, find_packages

setup(
name='fbpinn_wave',
version='0.1.0',
packages=find_packages(),
install_requires=['numpy', 'jax', 'fbpinns'],  # Add dependencies here, if any
author='Matías Di Liscia',
author_email='matias.diliscia99@gmail.com',
description='Herramientas de visualización para resolver la ecuación de onda 1D con FBPINNs',
)