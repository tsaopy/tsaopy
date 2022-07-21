"""Main submodule of the library."""
from tsaopy._f2pyauxmod import simulation, simulationv


#           Aux stuff, raises etc
class ModelInitException(Exception):
    """Custom exception for Event instance init."""

    def __init__(self, rtext):
        """Edit init."""
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Model object.')
        super().__init__(msg)


class Model:
    """Main object of the library."""

    def __init__(self, ode_coefs, events):
        """Do."""
        # run tests
        for i in ode_coefs:
            try:
                if len(ode_coefs[i]) < 3:
                    raise ModelInitException('ode coefs dict has less info '
                                             'than expected. Remember to add '
                                             'initial guess, prior, and index '
                                             'for each coefficient.')
                if len(ode_coefs[i]) > 3:
                    raise ModelInitException('ode coefs dict has more info '
                                             'than expected. Remember to only '
                                             'add initial guess, prior, and '
                                             'index for each coefficient.')
                ilen = len(ode_coefs[i][2])
                
            # general exception when we don't know what happened
            finally:
                raise ModelInitException('ode coefs dict has something wrong.')

        # a and b 1D vectors
        self.aindex = ode_coefs['a'][-1]
        self.alen, self.ashape = len(self.aindex), max(self.aindex)

        self.bindex = ode_coefs['b'][-1]
        self.blen, self.bshape = len(self.bindex), max(self.bindex)

        # f vector
        if 'f' not in ode_coefs:
            self.using_f = False
        elif 'f' in ode_coefs:
            self.using_f = True
            assert len

    def predict_sev(self, A, B, F, C, i):
        """Compute the prediction for a single event."""
        pass

    def predict(self, coords):
        """Compute the prediction for all events."""
        # 1D arrays coefs
        a_start, a_stop = 0, self.alen
        b_start, b_stop = a_stop, a_stop + self.blen
        f_start, f_stop = b_stop, b_stop + self.blen
        A, B, F = (coords[a_start:a_stop], coords[b_start:b_stop],
                   coords[f_start:f_stop])

        # matrix coefs
        c_start, c_stop = f_stop, f_stop + self.clen
        C = coords[c_start:c_stop]
        C.reshape(self.c_n, self.c_m)

        last_n = c_stop
        # iterate over all events
        for event in self.events:
            x0, v0 = coords[last_n], coords[last_n + 1]
            if event.use_ep:
                ep = coords[last_n + 2]
