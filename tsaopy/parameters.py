import sys
from tsaopy._bendtools import test_var_is_number

#                               PARAMETER CLASSES


class Fixed:
    """
    `tsaopy` fixed parameter class.

    This class represents a fixed `tsaopy` parameter object used to set up the
    `tsaopy` model.

    It's adviced against using the `Fixed` class since it gives the backend
    less freedom to optimize the fit. If the value is known it's advised to use
    a `Fitting` parameter with a very narrow prior instead, in most cases.
    """

    def __init__(self, value, ptype, index=None):
        """
        Parameters
        ----------
        value : int or float
            numerical value associated with the parameter. If parameter is
            Fixed it's the value assigned.
        ptype : str
            string with the values ep, x0, v0, log_fx, log_fv, a, b, c, or f.
            Used to identify if the parameter is an initial condition, an ODE
            coefficient, or another kind of parameter.
        index : int or touple, optional
            not necessary for ep, x0, v0, log_fx and log_fv ptypes (does
            nothing). Necessary for a, b, c, and f ptypes. If not set for those
            ptypes it will cause errors when building `tsaopy` models.

        Indexes should be assigned according to

                * for a and b parameters its the order of the term, eg:
                    b_2*x^2 => index = 2
                * for c parameters it is a touple with the order of x as first
                element and the order of x'=v as second element, eg:
                    c_21*x^2*x' => index = (2, 1)
                * for f parameters it is 1 for F_0, 2 for omega, and 3 for phi.
        """
        if not test_var_is_number(value):
            sys.exit("tsaopy params error: input value is not a number.")
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = True


class Fitting(Fixed):
    """
    `tsaopy` fitting parameter class.

    This class represents a fitting `tsaopy` parameter object used to set up the
    `tsaopy` model. In addition to the 'value', 'ptype', and 'index' arguments
    it is necessary to provide a prior in the form of a normalized PDF. Methods
    to easily define priors are given in the tools module.
    """

    def __init__(self, value, ptype, prior, index=None):
        """
        Parameters
        ----------
        value : int or float
            estimation of the unknown variable that will be supplied to the
            initial values of the chain.
        ptype : str
            string with the values ep, x0, v0, log_fx, log_fv, a, b, c, or f.
            Used to identify if the parameter is an initial condition, an ODE
            coefficient, or another kind of parameter.
        index : int or touple, optional
            not necessary for ep, x0, v0, log_fx and log_fv ptypes (does
            nothing). Necessary for a, b, c, and f ptypes. If not set for those
            ptypes it will cause errors when building `tsaopy` models.

        Indexes should be assigned according to

                * for a and b parameters its the order of the term, eg:
                    b_2*x^2 => index = 2
                * for c parameters it is a touple with the order of x as first
                element and the order of x'=v as second element, eg:
                    c_21*x^2*x' => index = (2, 1)
                * for f parameters it is 1 for F_0, 2 for omega, and 3 for phi.
        prior : callable
            a normalized probability density function represented as a
            callable. This object must return a number when being called with
            a number as argument.
        """
        super().__init__(value, ptype, index)
        self.fixed = False
        self.prior = prior
