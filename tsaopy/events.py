"""tsaopy Events submodule."""
import numpy as np


#           Aux stuff, raises etc
class EventInitException(Exception):
    """Custom exception for Event instance init."""

    def __init__(self, rtext):
        """Edit init."""
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Event object.')
        super().__init__(msg)


#           tsaopy scripts
class Event:
    """
    tsaopy Event class. Used to later instance tsaopy Model objects.

    The idea behind the Event class is to make it straightforward to fit the
    same equation of motion to different sets of measurements at the same time.

    Suppose we observed the same system several times with different initial
    conditions. Then we will set up an Event instace for each of our experiment
    runs. We will group them all together later when creating a Model class
    instance. With this Model instance we will be fitting all the different
    sets of measurements with the same equation of motion, but different
    initial conditions, equilibrium points, etc.
    """

    def __init__(self, x_data, t_data, params):
        """Instance parameters."""
        #           TO DO HERE: improve error handling ...
        # check x and t have the same lengths
        try:
            if not len(t_data) == len(x_data):
                raise ValueError('x_data and t_data have different lengths.')
        except Exception as exception:
            raise EventInitException("couldn't assert x_data and t_data "
                                     "have the same lengths.") from exception

        # check x and t have finite float values
        try:
            if not np.isfinite(x_data).all():
                raise ValueError('x_data has non finite values.')
            if not np.isfinite(t_data).all():
                raise ValueError('t_data has non finite values.')
        except Exception as exception:
            raise EventInitException("couldn't assert all values in x_array "
                                     "and t_array were finite numbers.") \
                                                            from exception

        # make sure that x0 and v0 are in params
        try:
            if 'x0' not in params:
                raise EventInitException('no x0 param when building Event.')
            if 'v0' not in params:
                raise EventInitException('no v0 param when building Event.')
        except Exception as exception:
            raise EventInitException("couldn't assert x0 and v0 keys were incl"
                                     "uded in params dict.") from exception

        # make sure that all params have 2 values
        for param in params:
            try:
                if len(params[param]) < 2:
                    raise EventInitException('missing value in a param.')
                elif len(params[param]) > 2:
                    raise EventInitException('too many values in a param.')
            except Exception as exception:
                raise EventInitException("couldn't assert all params have a "
                                         "guess and a prior.") from exception

        # check that all guesses return a positive value for the prior
        for param in params:
            try:
                x, p = params[param]
                if not np.isfinite(p(x)):
                    raise ValueError("initial guess for a param returned nan "
                                     "or inf for its prior.")
                if not p(x) > .0:
                    raise ValueError("initial guess for a param didn't return "
                                     "a positive value for its prior.")
            except Exception as exception:
                raise EventInitException("couldn't check that initial guess "
                                         "returns a positive prior value for "
                                         "some parameter.") from exception

        # do checks for ep
        if 'ep' in params:
            self.using_ep = True
        else:
            self.using_ep = False

        # do checks for fx
        if 'log_fx' in params:
            self.using_logfx = True
        else:
            self.using_logfx = False

        # do checks for fv
        if 'log_fv' in params:
            self.using_logfv = True
        else:
            self.using_logfv = False

        #       Define core attributes ~~~~
        self.datalen = len(x_data)
        self.dt = (t_data[-1] - t_data[0]) / (self.datalen - 1)

        def _compute_prediction(A, B, F, C, x0v0,
                                ep=0, log_fx=None, log_fv=None):
            return
