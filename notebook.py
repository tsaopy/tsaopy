import numpy as np
import backend as bend

# load data
data = np.loadtxt('experiment_data.txt')
data_t,data_x = data[:,0],data[:,1]
data_x_sigma = 0.2

# priors

x0_prior = bend.uniform_prior(0.7,1.3)
a1_prior = bend.uniform_prior(-5.0,5.0)
b1_prior = bend.uniform_prior(0.0,5.0)
    
# parameters

x0 = bend.FittingParameter(1.0,'x0',1,x0_prior)
v0 = bend.FixedParameter(0.0,'v0',1)
a1 = bend.FittingParameter(0.0, 'a', 1, a1_prior)
b1 = bend.FittingParameter(0.5,'b',1,b1_prior)

parameters = [x0,v0,a1,b1]

# set up main

model1 = bend.Model(parameters,data_t,data_x,data_x_sigma)

sampler,_,_,_ = model1.setup_sampler(200, 150, 200)
samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

# plots

label_list = model1.params_labels
bend.cornerplots(flat_samples,label_list)
bend.traceplots(samples,label_list)
bend.autocplots(flat_samples,label_list)
