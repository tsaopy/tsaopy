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

def _zero_df(t):
    return t*0

#           tsaopy scripts
class BaseModel:
    """
    tsaopy BaseModel class.

    This object condenses all necessary variables to set up the ODE according
    to the parameters provided plus the MCMC configuration to do the fitting.

    """

    def __init__(self, ode_coefs, events, driving_force=None):
        r"""
        Parameters
        ----------
        ode_coefs : dict
            dictionary containing the ode coefficients relevant in the model's
            ODE.

            Note: if the model considers driving force, all parameters
            must be included.

            Each key should be one of type of coefficients in the ODE ('a',
            'b', 'f', or 'c') and the value for each key should be a list with
            touples with the index and the prior for each coefficient.

            Indices are given by

            * the order of the term for 'a' and 'b' coefs. Eg: in \(b_2x^2\)
            the index of \(b_2\) coef is 2.
            * for 'f' coefs pass the index of each element in f_coords array
            starting at 1.
            * for 'c' coefs use a touple with two indices for the order of each
            factor. Eg: in \(c_{21}x^2\dot{x}\) index is (2, 1).
        events : list
            list containing all tsaopy Event objects to which the model will be
            fitted. Even if only one Event is used, pass it inside a list.
        driving_force : callable, optional
            A callable to compute the driving force. It must take two arguments
            , first a float for the time variable, and then an array with all
            the parameters other than the time variable. The default is
            None(uses \(F(t)=0\)). It must be mappable to numpy arrays.

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
 
            def dforce(t, f_params):
                F, w, p = f_params
                return F*np.sin(w*t + p)

            model1 = tsaopy.models.Model(ode_coefs=model1_ode_coefs,
                                         events=[event1, event2, event3],
                                         driving_force=dforce)
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

        # C matrix
        if 'c' in ode_coefs:
            ccoefs = ode_coefs['c']
            ccoefs.sort()
            self.odecoefs += ccoefs
            self.cind = [c[0] for c in ccoefs]
            self.paramslabels += [r'c_{' + str(i[0]) + str(i[1]) + '}'
                                  for i in self.cind]
            self.cdim = len(self.cind)
            self.cn = max([i[0] for i in self.cind])
            self.cm = max([i[1] for i in self.cind])
        else:
            self.cind = []
            self.cdim = 0
            self.cn, self.cm = 0, 0

        # f ~ driving force
        if 'f' not in ode_coefs and driving_force is None:
            self.df_function = _zero_df
            self.using_f = False
            self.fdim = 0
        elif 'f' not in ode_coefs and driving_force is not None:
            raise ModelInitException("driving_force was passed but there is "
                                     "no f key in ode_coefs.")
        elif 'f' in ode_coefs and driving_force is None:
            raise ModelInitException("f key was passed in ode_coefs but "
                                     "driving_force is None.")
        elif 'f' in ode_coefs and driving_force is not None:
            self.using_f = True
            self.df_function = driving_force
            fcoefs = ode_coefs['f']
            fcoefs.sort()
            self.fdim = len(fcoefs)
            self.odecoefs += fcoefs
            self.paramslabels += ['f' + str(i)
                                  for i in range(1, self.fdim + 1)]

        self.odecoefs_ndim = len(self.odecoefs)
        self.odepriors = [c[-1] for c in self.odecoefs]

        # finish param labels
        for i, event in enumerate(events):
            if event.using_tt:
                self.paramslabels.append(str(i+1) + ' - tt')
            if event.using_x0v0:
                self.paramslabels += [str(i+1) + ' - ' + s
                                      for s in ['x0', 'v0']]
            if event.using_ep:
                self.paramslabels.append(str(i+1) + ' - ep')
            if event.custom_ll_params:
                self.paramslabels += [str(i+1) + ' - ' + _ for _ in
                                      event.custom_ll_params_labels]

        # store events and final data
        self.events = events

        self.ndim = self.odecoefs_ndim
        for e in events:
            self.ndim += e.ndim

    def _ode_arrays(self, abc_coefs):
        alen, blen, cn, cm = self.alen, self.blen, self.cn, self.cm
        aind, bind, cind = self.aind, self.bind, self.cind
        adim, bdim = self.adim, self.bdim
        A, B, C = (np.zeros(alen), np.zeros(blen), np.zeros((cn, cm)))

        for i, ind in enumerate(aind):
            A[ind-1] = abc_coefs[i]
        last_n = adim

        for i, ind in enumerate(bind):
            B[ind-1] = abc_coefs[i + last_n]
        last_n += bdim

        for i, ind in enumerate(cind):
            n, m = ind
            C[n - 1, m - 1] = abc_coefs[i + last_n]

        return A, B, C

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
        abc_dim = self.adim + self.bdim + self.cdim
        f_dim = self.fdim
        abc_coefs, f_params, event_params = (coords[:abc_dim],
                                             coords[abc_dim: abc_dim + f_dim],
                                             coords[abc_dim + f_dim:])

        A, B, C = self._ode_arrays(abc_coefs)

        last_n = 0
        # iterate over all events
        for j, event in enumerate(self.events):
            if not j == i-1:
                last_n += event.ndim
            elif j == i-1:
                tt, x0v0, ep = 0, np.zeros(2), 0
                if event.using_tt:
                    tt = event_params[last_n]
                    last_n += 1
                if event.using_x0v0:
                    x0v0 = event_params[last_n:last_n + 2]
                    last_n += 2
                if event.using_ep:
                    ep = event_params[last_n]
                    last_n += 1
                return event._predict(A, B, C, self.df_function, f_params,
                                      tt, x0v0, ep)

    def _log_likelihood(self, coords):
        """Compute log likelihood."""
        abc_dim = self.adim + self.bdim + self.cdim
        f_dim = self.fdim
        abc_coefs, f_params, event_params = (coords[:abc_dim],
                                             coords[abc_dim: abc_dim + f_dim],
                                             coords[abc_dim + f_dim:])

        A, B, C = self._ode_arrays(abc_coefs)

        last_n = 0
        result = 0
        # iterate over all events
        for event in self.events:
            tt, x0v0, ep = 0, np.zeros(2), 0
            if event.using_tt:
                tt = event_params[last_n]
                last_n += 1
            if event.using_x0v0:
                x0v0 = event_params[last_n:last_n + 2]
                last_n += 2
            if event.using_ep:
                ep = event_params[last_n]
                last_n += 1
            if event.custom_ll_params:
                ll_params = event_params[last_n:last_n + event.cllp_ndim]
                last_n += event.cllp_ndim
            else:
                ll_params = None

            result += event._log_likelihood(A, B, C,
                                            self.df_function, f_params,
                                            tt, x0v0, ep,
                                            ll_params)
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

    def neg_ll(self, coords):
        """
        Compute the negative logarithmic likelihood.

        Compute the negative logarithmic likelihood for a set of parameter
        values for the defined model. This is best used when optimizing the
        initial values with an external optimizer.

        Parameters
        ----------
        coords : array
            values for each parameter. The elements must be passed in the same
            order than the parameters arg that was passed when initializing the
            model.

        Returns
        -------
        float
            value of `neg_ll` for the given parameter values.

        Examples
        -------
            f_to_minimize = my_model.neg_ll
            external_function_minimizer(f_to_minimize, *args)
        """
        return -self._log_likelihood(coords)
