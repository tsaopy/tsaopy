from setuptools import setup

setup(
   name='tsaopy',
   version='0.0.1-alpha',
   description=('Time Series by Anharmonic Oscillators is a Python library '
                'designed to analize oscillating time series by modelling '
                'them as anharmonic oscillators.'),
   author='Sofia A. Scozziero',
   author_email='sgscozziero@gmail.com',
   packages=['tsaopy'],
   url='https://tsaopy.github.io/',
   install_requires=['numpy','math','sys','matplotlib','multiprocessing',
                     'emcee','corner'], 
   scripts=[
           'tsaopy/f2pyauxmod'
           ]
)
