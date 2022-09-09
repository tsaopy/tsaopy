import os
import sys
import numpy.f2py

# cd to file directory
filedir = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
os.chdir(filedir)

# load fortran source
with open("fortransource.f90", "r") as f90_file:
    fortran_source = f90_file.read()

# compile fortran module
out = numpy.f2py.compile(fortran_source, modulename='_f2pyauxmod',
                         verbose=False, extension='.f90')

if not out == 0:
    sys.exit("Error when building tsaopy submodules, f2py compiler failed.")

os.rename('auxinit.py', '__init__.py')
os.chdir(cwd)

import tsaopy.events
import tsaopy.models
