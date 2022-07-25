"""Main submodule of the library."""
import numpy as np
import quickemcee as qmc


#           Aux stuff, raises etc
class ModelInitException(Exception):
    """Custom exception for Model instance init."""

    def __init__(self, rtext):
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Model object.')
        super().__init__(msg)


#           tsaopy scripts
class BaseModel:
    """
    tsaopy BaseModel class.

    This object condenses all necessary variables to set up the ODE according
    to the parameters provided plus the MCMC configuration to do the fitting.

    """

    def __init__(self, ode_coefs, events):
        r"""
        Parameters
        ----------
        ode_coefs : dict
            dictionary containing the ode coefficients relevant in the model's
            ODE.

            Note: if the model considers driving force, all three coeffients
            must be included.

            Each key should be one of type of coefficients in the ODE ('a',
            'b', 'f', or 'c') and the value for each key should be a list with
            touples with the index and the prior for each coefficient.

            Indices are given by

            * the order of the term for 'a' and 'b' coefs. Eg: in \(b_2x^2\)
            the index of \(b_2\) coef is 2.
            * for 'f' coefs, \(F_0\) is 1, \(\omega\) is 2, and \(\phi\) is 3.
            * for 'c' coefs use a touple with two indices for the order of each
            factor. Eg: in \(c_{21}x^2\dot{x}\) index is (2, 1).
        events : list
            list containing all tsaopy Event objects to which the model will be
            fitted. Even if only one Event is used, pass it inside a list.


        Examples
        --------
            event1 = tsaopy.events.Event(params1, t1, x1, x1_sigma)
            event2 = tsaopy.events.Event(params2, t2, x2, x2_sigma,
                                         v2, v2_sigma)
            event3 = tsaopy.events.Event(params3, t3, x3, x3_sigma)

            model1_ode_coefs = {'a': [(1, a1_prior), (2, a2_prior)],
                                'b': [(1, b1_prior)],
                                'f': [(1, F_prior), (2, w_prior),
                                      (3, p_prior)],
                                'c': [((2, 1), c21_prior)]}

            model1 = tsaopy.models.Model(ode_coefs=model1_ode_coefs,
                                         events=[event1, event2, event3])
        """
        # run tests
        for i in ode_coefs:
            try:
                ilist = ode_coefs[i]
                # check that every parameter has 2 elements in its touple
                for coef in ilist:
                    if len(coef) < 2:
                        raise ModelInitException('some ODE coef is missing '
                                                 'parameters.')
                    elif len(coef) > 2:
                        raise ModelInitException('some ODE coef has too many '
                                                 'parameters.')
                    # check that for every parameter p(x) > 0
                    else:
                        (_, p), x = coef, np.random.normal(.0, 100.0)
                        if p(x) < .0:
                            raise ModelInitException("some ODE prior returned "
                                                     "a negative value when "
                                                     "called with random "
                                                     "float.")
            # general exception
            except Exception as exception:
                raise ModelInitException('') from exception

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
            if not len(ode_coefs['f']) == 3:
                raise ModelInitException("f key provided in ode_coefs dict but"
                                         " its value's length is not 3.")
            self.using_f = True
            fcoefs = ode_coefs['f']
            fcoefs.sort()
            self.odecoefs += fcoefs
            self.paramslabels += [r'F_0', r'\omega', r'\phi']

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

    def setup_mcmc_model(self):
        """
        Build `quickemcee` model object.

        Returns
        -------
        quickemcee model object
            `quickemcee` model instance that can be used to run MCMC chains.
            See `quickemcee` docs for this class usage.

        Examples
        --------
            tsaopymodel = tsaopy.models.Model(ode_coefs_dict, events_list)
            qmcmodel = tsaopymodel.setup_mcmc_model()
            sampler = qmcmodel.run_chain(100, 50, 100)
        """
        return qmc.core.LPModel(self.ndim, self._log_probability)

    def event_predict(self, i, coords):
        """
        Compute the prediction of the model for the i-eth event.

        This method has the purpose of plotting the prediction of the model for
        one of the events it was fitted to. Used for plotting with the results.

        Parameters
        ----------
        i : int
            An integer 1, 2, 3.. indicating the position of the event to
            simulate in the events list provided to the Model instace. Don't
            start at 0.'
        coords : array
            an array of length ndim(number of ode coefs plus number of params
            for each model, total number of parameters to fit).

        Returns
        -------
        array
            array of the same shape as the x_data attribute in the i-eth event.

        """
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
                last_n += 1
            if event.using_logfv:
                last_n += 1
            if j == i-1:
                return event._predict(A, B, F, C, x0v0, ep)
