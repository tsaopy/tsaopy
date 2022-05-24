import sys
sys.path.append('../..')
import numpy as np
import tsaopy

# load data
data = np.loadtxt('experiment_data.txt')
data_t,data_x = data[:,0],data[:,1]
data_x_sigma = 0.15

# priors

x0_prior = tsaopy.tools.uniform_prior(0.7,1.3)
v0_prior = tsaopy.tools.uniform_prior(-1.0,1.0)
a1_prior = tsaopy.tools.uniform_prior(-5.0,5.0)
b1_prior = tsaopy.tools.uniform_prior(0.0,5.0)
    
# parameters

x0 = tsaopy.parameters.FittingParameter(1.0,'x0',1,x0_prior)
v0 = tsaopy.parameters.FittingParameter(0.0,'v0',1,v0_prior)
a1 = tsaopy.parameters.FittingParameter(0.0, 'a', 1, a1_prior)
b1 = tsaopy.parameters.FittingParameter(0.5,'b',1,b1_prior)

parameters = [x0,v0,a1,b1]

# model 1 (no velocity)

model1 = tsaopy.models.Model(parameters,data_t,data_x,data_x_sigma)

sampler,_,_,_ = model1.setup_sampler(200, 300, 1200)
samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

label_list = model1.params_labels
tsaopy.tools.cornerplots(flat_samples,label_list)
tsaopy.tools.traceplots(samples,label_list)
tsaopy.tools.autocplots(flat_samples,label_list)

solutions = [np.mean(flat_samples[:,_]) for _ in range(len(parameters))]
model1.plot_simulation(solutions)
