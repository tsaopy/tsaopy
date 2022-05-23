import numpy as np
import backend as bend

# load data
data = np.loadtxt('experiment_data.txt')
data_t,data_x,data_v = data[:,0],data[:,1],data[:,2]
data_x_sigma,data_v_sigma = 0.2,0.15

# priors

x0_prior = bend.uniform_prior(0.7,1.3)
v0_prior = bend.uniform_prior(0.3,0.7)
b1_prior = bend.normal_prior(0.0,10.0)
b3_prior = bend.normal_prior(0.0,10.0)
    
# parameters

x0 = bend.FittingParameter(1.0,'x0',1,x0_prior)
v0 = bend.FittingParameter(0.5,'v0',1,v0_prior)
b1 = bend.FittingParameter(0.0,'b',1,b1_prior)
b3 = bend.FittingParameter(0.0,'b',3,b3_prior)

parameters = [x0,v0,b1,b3]

# model 2

model2 = bend.VelocityModel(parameters,data_t,data_x,data_v,
                            data_x_sigma,data_v_sigma)
neg_ll = model2.neg_ll


from scipy.optimize import differential_evolution
bounds = [(0.7,1.3),(0.3,0.7),(-30,30),(-30,30)]
diffev_solution = differential_evolution(neg_ll,bounds=bounds,popsize=50,
                              mutation=(1.0,1.9),maxiter=2000,workers=6)

new_initvals = diffev_solution.x
model2.update_initvals(new_initvals)

sampler,_,_,_ = model2.setup_sampler(300, 100, 500)
samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

label_list = model2.params_labels
bend.cornerplots(flat_samples,label_list)
bend.traceplots(samples,label_list)
bend.autocplots(flat_samples,label_list)

solutions = [np.mean(flat_samples[:,_]) for _ in range(len(parameters))]
model2.plot_simulation(solutions)
