"""Main submodule of the library."""
from tsaopy._f2pyauxmod import simulation, simulationv

class Model:
    """Main object of the library."""

    def __init__(self, ode_coefs, events):
        """Do."""
        pass

    def predict_sev(self, A, B, F, C, i):
        """Compute the prediction for a single event."""

        use_ep = self.events[i].use_ep
        if self.events[i].use_ep a

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
                ep = 
