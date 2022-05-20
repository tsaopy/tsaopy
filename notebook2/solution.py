import numpy as np
import backend as bend

# load data
data = np.loadtxt('experiment_data.txt')
data_t,data_x,data_v = data[:,0],data[:,1],data[:,2]
data_x_sigma,data_v_sigma = 0.4,0.5

# priors

x0_prior = bend.uniform_prior(1.5,2.5)
v0_prior = bend.uniform_prior(-1.0,1.0)
a1_prior = bend.uniform_prior(-5.0,5.0)
a2_prior = bend.uniform_prior(-5.0,5.0)
b1_prior = bend.uniform_prior(0.0,5.0)
b2_prior = bend.uniform_prior(-5.0,5.0)
c11_prior = bend.uniform_prior(-5.0,5.0)
c12_prior = bend.uniform_prior(-5.0,5.0)
c21_prior = bend.uniform_prior(-5.0,5.0)
c22_prior = bend.uniform_prior(-5.0,5.0)


# parameters

x0 = bend.FittingParameter(2.0,'x0',1,x0_prior)
v0 = bend.FittingParameter(0.0,'v0',1,v0_prior)
a1 = bend.FittingParameter(0.0, 'a', 1, a1_prior)
a2 = bend.FittingParameter(0.0, 'a', 2, a2_prior)
b1 = bend.FittingParameter(0.5,'b',1,b1_prior)
b2 = bend.FittingParameter(0.0,'b',2,b2_prior)
c11 = bend.FittingParameter(0.0,'c',(1,1),c11_prior)
c12 = bend.FittingParameter(0.0,'c',(1,2),c12_prior)
c21 = bend.FittingParameter(0.0,'c',(2,1),c21_prior)
c22 = bend.FittingParameter(0.0,'c',(2,2),c22_prior)


parameters = [x0,v0,a1,a2,b1,b2,c11,c12,c21,c22]

# model 2 (velocity)

model2 = bend.VelocityModel(parameters,data_t,data_x,data_v,
                            data_x_sigma,data_v_sigma)

sampler,_,_,_ = model2.setup_sampler(200, 300, 300)
samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

label_list = model2.params_labels
bend.cornerplots(flat_samples,label_list)
bend.traceplots(samples,label_list)
bend.autocplots(flat_samples,label_list)
