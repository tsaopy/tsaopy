import sys
from tsaopy.bendtools import test_var_is_number

#                               PARAMETER CLASSES


class Fixed:
    """
    This class creates a fixed tsaopy parameter object used to set up the tsaopy model. It will
    tell the backend which term of the ODE this coefficient belongs to, and how much its value is.

    I advice against using the Fixed class since it gives the backend less freedom to optimize the
    fit. If you are very confident in a certain parameter value you can use a Fitting parameter
    with a very narrow prior instead.
    """

    def __init__(self, value, ptype, index):
        if not test_var_is_number(value):
            sys.exit("tsaopy params error: input value is not a number.")
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = True


class Fitting(Fixed):
    """
    This class creates a fitting tsaopy parameter object used to set up the tsaopy model. It will
    tell the backend which term of the ODE this coefficient belongs to, and what your initial
    estimate of its value is. In addition to the 'value', 'ptype', and 'index' arguments you need
    to provide a prior in the form of a normalized PDF. Methods to easily define priors are given
    in the tools module.
    """

    def __init__(self, value, ptype, index, prior):
        super().__init__(value, ptype, index)
        self.fixed = False
        self.prior = prior
