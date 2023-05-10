from setuptools import setup, find_packages

setup(
    name='calibrationframework',
    version='0.1.2',
    url='https://github.com/lorenzofamiglini/CalFram',
    author='Lorenzo Famiglini',
    author_email='lorenzofamiglini@gmail.com',
    description='A comprehensive framework for assessing calibration in machine and deep learning models',
    packages=find_packages(),
    install_requires=[
    'pandas',
    'numpy',
    'matplotlib',
    'scikit-learn',
    'ipdb',
]
)
