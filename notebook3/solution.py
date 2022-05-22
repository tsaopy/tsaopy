import numpy as np
import backend as bend

# load data
data = np.loadtxt('experiment_data.txt')
data_t,data_x = data[:,0],data[:,1]
data_x_sigma = 200

# priors

x0_prior = bend.normal_prior(-148,31)
v0_prior = bend.normal_prior(1920,380.0)
a1_prior = bend.normal_prior(6.0,5.0)
b1_prior = bend.normal_prior(90.0,20.0)
f_prior = bend.normal_prior(12000.0,1000.0)
w_prior = bend.normal_prior(9.40,0.01)
p_prior = bend.normal_prior(0.63,0.31)


# parameters

x0 = bend.FittingParameter(-148.0,'x0',1,x0_prior)
v0 = bend.FittingParameter(1920.0,'v0',1,v0_prior)
a1 = bend.FittingParameter(6.0, 'a', 1, a1_prior)
b1 = bend.FittingParameter(90.0,'b',1,b1_prior)
f = bend.FittingParameter(12000.0,'f',1,f_prior)
w = bend.FittingParameter(9.40,'f',2,w_prior)
p = bend.FittingParameter(0.63,'f',3,p_prior)

parameters = [x0,v0,a1,b1,f,w,p]

# model 1 (no velocity)

model1 = bend.Model(parameters,data_t,data_x,data_x_sigma)

sampler,_,_,_ = model1.setup_sampler(1000, 1500, 500)

samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)
label_list = model1.params_labels

bend.cornerplots(flat_samples,label_list)
bend.traceplots(samples,label_list)
bend.autocplots(flat_samples,label_list)

solutions = [np.mean(flat_samples[:,_]) for _ in range(len(parameters))]
model1.plot_simulation(solutions)
