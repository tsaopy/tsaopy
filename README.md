We are working on a Github page with more info at: https://tsaopy.github.io/.

# Files

## root

### backend
Script containing main algorithms.

### integrator_f2py
Script containing numerical integrators written in Fortran and wrapped to be used in Python by the backend. User should only be running this script once to build the `f2py` module that backend uses, and then leave this file alone. We will not be giving details of how this or the Fortran code work, as this escapes the original purpose of making it as simple as possible to the user.

## notebooks

### maketestdata
Script that produces a simulated time series from a differential equation defined by the user. 

### solution
Script with a `TSAOpy` solution of the parameter fitting problem for the time series made in maketestdata. It's a shortened and concise version of the notebooks we will be explaining in the Gihub page. It should give a good solution as well as plots to analyze the results by just running the full script. 

1. Notebook 1 proposes a simple linear damped oscillator.
2. Notebook 2 proposes a Van der Pol oscillator, and includes fitting to the $(t,v)$ data on addition to the already in use $(t,x)$ data.
