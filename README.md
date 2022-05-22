We are working on a Github page with more info at: https://tsaopy.github.io/.

We just set up a Zenodo DOI for referencing the repository

[![DOI](https://zenodo.org/badge/427913804.svg)](https://zenodo.org/badge/latestdoi/427913804)

# Files

## root

### backend
Script containing main algorithms.

### integrator_f2py
Script containing numerical integrators written in Fortran and wrapped to be used in Python by the backend. User should only be running this script once to build the `f2py` module that backend uses, and then leave this file alone. We will not be giving details of how this or the Fortran code work, as this escapes the original purpose of making it as simple as possible for the user.

## notebooks

### maketestdata
Script that produces a simulated time series from a differential equation defined by the user. 

### solution
Script with a `TSAOpy` solution of the parameter fitting problem for the time series made in maketestdata. It's a shortened and concise version of the notebooks we will be explaining in the Gihub page. It should give a good solution as well as plots to analyze the results by just running the full script. 

More info about the notebooks in [the Github page](https://tsaopy.github.io/).
