"""tsaopy Events submodule."""


#           Aux stuff, raises etc
class EventInitException(Exception):
    """Custom exception for Event instance init."""

    def __init__(self, rtext):
        """Edit init."""
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Event object.')
        super().__init__(msg)


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
        # make sure that x0 and v0 are here
        if 'x0' not in params:
            raise EventInitException('no x0 parameter when building Event.')
        if 'v0' not in params:
            raise EventInitException('no v0 parameter when building Event.')

        # make sure that all params have 2 values (guess and prior)
        for param in params:
            try:
                if len(params[param]) < 2:
                    raise EventInitException('missing value in a param.')
                elif len(params[param]) > 2:
                    raise EventInitException('too many values in a param.')
            finally:
                raise EventInitException("can't check if all given parameters "
                                         "have both a guess and a prior.")

        # here we check that all guesses return a positive value for the prior
        for param in params:
            try:
                x, p = params[param]
                if not p(x) > .0:
                    raise EventInitException("initial guess for a param didn't"
                                             " return positive value for its"
                                             " prior.")
            finally:
                raise EventInitException("couldn't check that initial guess "
                                         "returns a positive value for some "
                                         "parameter.")

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
