"""Main submodule of the library."""
import numpy as np
import quickemcee as qemc


#           Aux stuff, raises etc
class ModelInitException(Exception):
    """Custom exception for Model instance init."""

    def __init__(self, rtext):
        """Edit init."""
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Model object.')
        super().__init__(msg)


#           tsaopy scripts
class Model:
    """Main object of the library."""

    def __init__(self, ode_coefs, events):
        """Do."""
        # run tests
        for i in ode_coefs:
            try:
                ilist = ode_coefs[i]
                # check that every parameter has 3 elements in its touple
                for coef in ilist:
                    if len(coef) < 3:
                        raise ModelInitException('some ODE coef is missing '
                                                 'parameters.')
                    if len(coef) > 3:
                        raise ModelInitException('some ODE coef has too many '
                                                 'parameters.')
                    # check that for every parameter p(x) > 0
                    else:
                        _, x, p = coef
                        if not p(x) > .0:
                            raise ModelInitException('some ODE coef initial '
                                                     'guess does not return '
                                                     'a positive value for its'
                                                     ' prior.')
            # general exception
            except Exception as exception:
                raise ModelInitException('ode coefs dict has something wrong.'
                                         ) from exception

        self.odecoefs = []
        self.paramslabels = []
        # a and b 1D vectors
        if 'a' in ode_coefs:
            acoefs = ode_coefs['a']
            acoefs.sort()
            self.odecoefs += acoefs
            self.aind = [c[0] for c in acoefs]
            self.paramslabels += ['a' + str(i) for i in self.aind]
            self.adim = len(self.aind)
            self.alen = max(self.aind)
        else:
            self.aind = []
            self.adim = 0
            self.alen = 0

        if 'b' in ode_coefs:
            bcoefs = ode_coefs['b']
            bcoefs.sort()
            self.odecoefs += bcoefs
            self.bind = [c[0] for c in bcoefs]
            self.paramslabels += ['b' + str(i) for i in self.bind]
            self.bdim = len(self.bind)
            self.blen = max(self.bind)
        else:
            self.bind = []
            self.bdim = 0
            self.blen = 0

        # f vector
        if 'f' not in ode_coefs:
            self.using_f = False
        elif 'f' in ode_coefs:
            assert len(ode_coefs['f']) == 3, ('Error building tsaopy model: f '
                                              'key provided but len not 3.')
            self.using_f = True
            self.paramslabels += [r'F_0', r'\omega', r'\phi']
            self.odecoefs_ndim += 3

        # C matrix
        if 'c' in ode_coefs:
            ccoefs = ode_coefs['c']
            ccoefs.sort()
            self.odecoefs += ccoefs
            self.cind = [c[0] for c in ccoefs]
            self.paramslabels += [r'c_{' + str(i[0]) + str(i[1]) + '}'
                                  for i in self.bind]
            self.cdim = len(self.cind)
            self.cn = max([i[0] for i in self.cind])
            self.cm = max([i[1] for i in self.cind])
        else:
            self.cind = []
            self.cdim = 0
            self.cn, self.cm = 0, 0

        self.odecoefs_ndim = len(self.odecoefs)
        self.odepriors = [c[-1] for c in self.odecoefs]

        # finish param labels & ini guess
        for i, event in enumerate(events):
            self.paramslabels += [str(i+1) + ' - ' + s
                                  for s in ['x0', 'v0']]
            if event.using_ep:
                self.paramslabels.append(str(i+1) + ' - ep')
            if event.using_logfx:
                self.paramslabels.append(str(i+1) + ' - log_fx')
            if event.using_logfv:
                self.paramslabels.append(str(i+1) + ' - log_fv')

        # store events and final data
        self.events = events

        self.ndim = self.odecoefs_ndim
        for e in events:
            self.ndim += e.ndim

    def _ode_arrays(self, ode_coefs):
        alen, blen, cn, cm = self.alen, self.blen, self.cn, self.cm
        aind, bind, cind = self.aind, self.bind, self.cind
        adim, bdim = self.adim, self.bdim
        A, B, F, C = (np.zeros(alen), np.zeros(blen), np.zeros(3),
                      np.zeros((cn, cm)))

        for i, ind in enumerate(aind):
            A[ind-1] = ode_coefs[i]
        last_n = adim

        for i, ind in enumerate(bind):
            B[ind-1] = ode_coefs[i + last_n]
        last_n += bdim

        if self.using_f:
            F = ode_coefs[last_n: last_n + 3]
            last_n += 3

        for i, ind in enumerate(cind):
            n, m = ind
            C[n - 1, m - 1] = ode_coefs[i + last_n]

        return A, B, F, C

    def _log_likelihood(self, coords):
        """Compute log likelihood."""
        odecoefs_ndim = self.odecoefs_ndim
        ode_coefs, event_params = (coords[:odecoefs_ndim],
                                   coords[odecoefs_ndim:])

        A, B, F, C = self._ode_arrays(ode_coefs)

        last_n = 0
        result = .0
        # iterate over all events
        for event in self.events:
            x0v0 = event_params[last_n:last_n + 2]
            last_n += 2
            if event.using_ep:
                ep = event_params[last_n]
                last_n += 1
            else:
                ep = None
            if event.using_logfx:
                logfx = event_params[last_n]
                last_n += 1
            else:
                logfx = None
            if event.using_logfv:
                logfv = event_params[last_n]
                last_n += 1
            else:
                logfv = None

            result += event._log_likelihood(A, B, F, C, x0v0,
                                            ep, logfx, logfv)

        return result

    def _log_prior(self, coords):
        """Compute log prior."""
        odecoefs_ndim = self.odecoefs_ndim
        odecoords = coords[:odecoefs_ndim]
        result = 1
        for i, p in enumerate(self.odepriors):
            prob = p(odecoords[i])
            if prob <= .0:
                return -np.inf
            result *= prob
        result = np.log(result)

        last_n = odecoefs_ndim
        # iterate over all events
        for event in self.events:
            event_ndim = event.ndim
            event_coords = coords[last_n:last_n + event_ndim]
            last_n += event_ndim
            result += event._log_prior(event_coords)

        return result

    def _log_probability(self, coords):
        """Compute log probability."""
        lp = self._log_prior(coords)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(coords)

    def event_predict(self, i, coords):
        """Do."""
        odecoefs_ndim = self.odecoefs_ndim
        ode_coefs, event_params = (coords[:odecoefs_ndim],
                                   coords[odecoefs_ndim:])

        A, B, F, C = self._ode_arrays(ode_coefs)

        last_n = 0
        # iterate over events
        for j, event in enumerate(self.events):
            x0v0 = event_params[last_n:last_n + 2]
            last_n += 2
            if event.using_ep:
                ep = event_params[last_n]
                last_n += 1
            else:
                ep = None
            if event.using_logfx:
                logfx = event_params[last_n]
                last_n += 1
            else:
                logfx = None
            if event.using_logfv:
                logfv = event_params[last_n]
                last_n += 1
            else:
                logfv = None        
            if j == i-1:
                return event.predict(A, B, F, C, x0v0, ep)

    def setup_mcmc_model(self):
        """
        Build `quickemcee` model object.

        Returns
        -------
        quickemcee model object
            `quickemcee` model instance that can be used to run MCMC chains.

        """
        return qemc.core.LPModel(self.ndim, self._log_probability)
