import sys
import numpy as np
from multiprocessing import cpu_count, Pool
from matplotlib import pyplot as plt
import emcee
import corner
from oscadsf2py import simulation

# auxiliary functions

def flatten_seq(array):
    result, shapes_info = [],[]
    for array_element in array:
        aux = np.array(array_element)
        shapes_info.append(aux.shape)
        aux_flattened = np.hstack(aux)
        result = result + list(aux_flattened)
    return np.array(result), shapes_info

def reshape_matrix(elements, shape):
    if not len(shape)==2:
        sys.exit('Error: array to reshape is not a matrix.')
    ilen,jlen = shape
    result = np.zeros(shape)
    for i in range(ilen):
        for j in range(jlen):
            result[i,j] = elements[j+i*jlen]
    return result

def deflatten_seq(flat_array, flat_info):
    result, current_index = [],0
    for step_info in flat_info:
        index_dif = np.prod(step_info)
        step_elements = flat_array[current_index:current_index+index_dif]
        current_index = current_index + index_dif
        if len(step_info)==2:
            result.append(reshape_matrix(step_elements,step_info))
        elif len(step_info)==1:
            result.append(step_elements)
    return result

# main program

# parameter classes
class FixedParameter:
    def __init__(self,value,ptype,index):
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = True
        
class FittingParameter:
    def __init__(self,value,ptype,index,prior):
        self.value = value
        self.ptype = ptype
        self.index = index
        self.fixed = False
        self.prior = prior

# aux functions 2

def param_ptype_shape(params,test_ptype):
    if test_ptype == 'x0' or test_ptype == 'v0':
        sys.exit('Error: test ptype is x0 or v0.')
    indexes = []
    for elem in params:
        if elem.ptype == test_ptype:
            indexes.append(elem.index)
    if test_ptype=='c' and not len(indexes)==0:
        arraux = np.array(indexes)
        return max(arraux[:,0]),max(arraux[:,1])
    elif test_ptype=='c' and len(indexes)==0:
        return (0,0)
    elif (test_ptype=='a' or test_ptype=='b' or test_ptype=='f') and not\
        len(indexes)==0:
        return (max(indexes),)
    elif (test_ptype=='a' or test_ptype=='b' or test_ptype=='f') and \
        len(indexes)==0:
        return (0,)
    else:
        sys.exit('Error: test ptype is not a, b, c, or f.')
    pass
        
def params_array_shape(params):
    ptypes_set = []
    for _ in params:
        ptypes_set.append(_.ptype)
    ptypes_set = set(ptypes_set)
    
    if (not 'x0' in ptypes_set) or (not 'v0' in ptypes_set):
        sys.exit('Error: initial conditions are not properly defined.')

    p_shape_array = [(1,),(1,)]
    
    for _ in ['a','b','c']:
        p_shape_array.append(param_ptype_shape(params,_))
        
    p_shape_array.append((3,))
    
    return p_shape_array

def param_names(param):
    ptype,index = param.ptype,param.index
    if ptype == 'c' and len(index) == 2:
        return 'c'+str(index[0])+str(index[1])
    elif ptype == 'a' or ptype == 'b':
        return ptype+str(index)
    elif ptype == 'f' and index==1:
        return 'F'
    elif ptype == 'f' and index==2:
        return 'w'
    elif ptype == 'f' and index==3:
        return 'p'
    elif ptype == 'x0' or ptype == 'v0':
        return ptype
    else:
        sys.exit('Error naming parameters.')
    pass

def param_cindex(param):
    if param.ptype=='x0':
        return 0,0
    elif param.ptype=='v0':
        return 0,1
    elif param.ptype=='a':
        return 1,param.index-1
    elif param.ptype=='b':
        return 2,param.index-1
    elif param.ptype=='c':
        q = param.index
        return 3,(q[0]-1,q[1]-1)
    elif param.ptype=='f':
        return 4,param.index-1

def fitparams_coord_info(fparams):
    indexes,labels = [],[]
    for _ in fparams:
        indexes.append( param_cindex(_) )
        labels.append( param_names(_) )
    return indexes,labels
    
# model class

class Model:
    def __init__(self,parameters,t_data,x_data,
                 x_unc,tsplit=4):
        self.parameters = parameters
        self.t_data = t_data
        self.x_data = x_data
        self.x_unc = x_unc
        
        self.t0 = t_data[0]
        self.tsplit = tsplit
        self.dt = (t_data[1] - self.t0)/self.tsplit
        
        self.params_to_fit = [_ for _ in parameters if not _.fixed]
        self.ndim = len(self.params_to_fit)
        self.ptf_ini_values = [p.value for p in self.params_to_fit]
        self.ptf_info,self.params_labels = \
            fitparams_coord_info(self.params_to_fit)
        self.priors_array = [_.prior for _ in self.params_to_fit]
        
        self.fixed_values = { param_names(elem) : elem.value for elem
                             in self.parameters if elem.fixed}
        self.datalen = len(x_data)
        
        self.parray_shape = params_array_shape(self.parameters)
        self.alens = self.parray_shape[2][0], \
            self.parray_shape[3][0],self.parray_shape[4][0], \
            self.parray_shape[4][1]
    
    # simulations
    
    def setup_simulation_arrays(self,coords):
        
        na,nb,cn,cm, = self.alens
        x0v0,A,B,C,F = np.zeros(2),np.zeros(na),np.zeros(nb),\
            np.zeros((cn,cm)),np.zeros(3)
            
        for _ in self.parameters:
            if _.ptype == 'x0':
                x0v0[0] = _.value
            elif _.ptype == 'v0':
                x0v0[1] = _.value
            elif _.ptype == 'a':
                A[_.index-1] = _.value
            elif _.ptype == 'b':
                B[_.index-1] = _.value
            elif _.ptype == 'c':
                q = _.index
                C[(q[0]-1,q[1]-1)] = _.value
            elif _.ptype == 'f':
                F[_.index-1] = _.value
        
        results = [x0v0,A,B,C,F]
        ptf_index_info = self.ptf_info
        
        for q in ptf_index_info:
            i = ptf_index_info.index(q)
            results[q[0]][q[1]] = coords[i]
            
        return results
    
    def predict(self,coords):
        
        dt,tsplit,datalen = self.dt,self.tsplit,self.datalen
        na,nb,cn,cm, = self.alens
            
        x0v0_simu, A_simu, B_simu, C_simu, F_simu = \
            self.setup_simulation_arrays(coords)
        
        return simulation(x0v0_simu, A_simu, B_simu, C_simu, F_simu, dt,
                          tsplit*datalen, na, nb, cn, cm)[::tsplit]
    
    # mcmc stuff
        
    def log_prior(self,coefs):
        result = 1
        for i in range(self.ndim):
            prob = self.priors_array[i](coefs[i])
            if prob <= 0:
                return -np.inf
            else:
                result = result*prob
        return np.log(result)
        
    def log_likelihood(self,coefs):
        prediction = self.predict(coefs)
        if not np.isfinite(prediction[-1]):
              return -np.inf
        ll = -0.5*np.sum(((prediction-self.x_data)/self.x_unc)**2)
        return ll
    
    def log_probability(self,coefs):
        lp = self.log_prior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(coefs)
    
    def setup_sampler(self, n_walkers, burn_iter, main_iter, 
                      cores=(cpu_count()-2)):

        p0 = [self.ptf_ini_values + 1e-7 * np.random.randn(self.ndim) 
                  for i in range(n_walkers)]
        
        with Pool(processes=cores) as pool:
             sampler = emcee.EnsembleSampler(n_walkers, self.ndim,
                                        self.log_probability, pool=pool)

             print("Running burn-in...")
             p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
             sampler.reset()

             print('')
             print("Running production...")
             pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

             return sampler, pos, prob, state
    
    # tools
    
    def neg_ll(self,coords):
        return -self.log_probability(coords)
    
    def plot_measurements(self,figsize=(7,5),dpi=100):
        plt.figure(figsize=figsize,dpi=dpi)
        plt.scatter(self.t_data,self.x_data,color='tab:red',s=5.0)
        plt.show()
        pass
    
    def plot_simulation(self,coords,figsize=(7,5),dpi=100):
        plt.figure(figsize=figsize,dpi=dpi)
        plt.scatter(self.t_data,self.x_data,color='black',s=5.0)
        plt.plot(self.t_data,self.predict(coords),color='tab:red')
        plt.show()
        pass

    
# extra

def make_cornerplots(flat_samples,labels_list):
    dim = len(labels_list)
    sample_truths = [np.mean(flat_samples[:, _]) for _ in range(dim)]

    fig = corner.corner(flat_samples, labels=labels_list,
                        quantiles=(0.16, 0.84), show_titles=True,
                        title_fmt='.3g', truths=sample_truths,
                        truth_color='tab:red')
    
    plt.show()
