import sys
from tsaopy.bendtools import test_var_is_number

#                               PARAMETER CLASSES


class Fixed:
    """

    This class represents a fixed tsaopy parameter object used to set up the
    tsaopy model. It will tell the backend which term of the ODE this
    coefficient belongs to, and how much its value is.

    I advice against using the Fixed class since it gives the backend less
    freedom to optimize the fit. If you are very confident in a certain
    parameter value you can use a Fitting parameter with a very narrow prior
    instead.

    """

    def __init__(self, value, ptype, index):
        """

        Initialice the instance.

        Args:
        value(int or float):  numerical value associated with the parameter.
        If parameter is Fixed it's the value assigned.
        ptype(str): string with the values x0, v0, a, b, c, or f. Used to
        identify if the parameter is an initial condition, or an ODE
        coefficient.
        index(int of int touple):
            - takes value 1 for x0 and v0 parameters
            - for a and b parameters its the order of the term,
                eg: b_2*x^2 => index = 2
            - for c parameters it is a touple with the order of x as first
            element and the order of x'=v as second element,
                eg: c_21*x^2*x' => index = (2, 1)
            - for f parameters it is 1 for F_0, 2 for omega, and 3 for phi.

        Returns:
        tsaopy parameter object instance.

        """
        if not test_var_is_number(value):
            sys.exit("tsaopy params error: input value is not a number.")
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = True


class Fitting(Fixed):
    """

    This class creates a fitting tsaopy parameter object used to set up the
    tsaopy model. It will tell the backend which term of the ODE this
    coefficient belongs to, and what your initial estimate of its value is.
    In addition to the 'value', 'ptype', and 'index' arguments you need to
    provide a prior in the form of a normalized PDF. Methods to easily define
    priors are given in the tools module.

    """

    def __init__(self, value, ptype, index, prior):
        """

        Initialice the instance.

        Args:
        value(int or float): estimation of the unknown variable that will be
        supplied to the initial values of the chain.
        ptype(str): string with the values x0, v0, a, b, c, or f. Used to
        identify if the parameter is an initial condition, or an ODE
        coefficient.
        index(int of int touple):
            - takes value 1 for x0 and v0 parameters
            - for a and b parameters its the order of the term,
                eg: b_2*x^2 => index = 2
            - for c parameters it is a touple with the order of x as first
            element and the order of x'=v as second element,
                eg: c_21*x^2*x' => index = (2, 1)
            - for f parameters it is 1 for F_0, 2 for omega, and 3 for phi.
        prior(callable): a normalized probability density function represented
        as a callable. This object must return a number when being called with
        a number as argument.

        Returns:
        tsaopy parameter object instance.

        """
        super().__init__(value, ptype, index)
        self.fixed = False
        self.prior = prior
