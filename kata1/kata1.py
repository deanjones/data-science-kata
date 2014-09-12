# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

true_b0 = 5
true_b1 = 3
true_e_sigma = 2
true_e = np.random.normal(true_e_mean, true_e_sigma, 100)
x = np.random.uniform(0, 20, 100)
data = pd.DataFrame({'x' : x, 'error' : true_e})
data['y'] = (data['x'] * true_b1) + true_b0 + data['error']
#fig = plt.figure()
#plot1 = fig.add_subplot(1,1,1)
#plot1.scatter(data['x'], data['y'])
#plt.show()

b0 = pm.Normal('b0', 0, 0.0001)
b1 = pm.Normal('b1', 0, 0.0001)
sigma = pm.Uniform('sigma', 0, 100)
x_obs = pm.Uniform('x_obs', 0, 20, value=data.x.values, observed=True)
@pm.deterministic
def pred(b0=b0, b1=b1, x_obs=x_obs):
    return x_obs * b1 + b0
y = pm.Normal('y', pred, sigma, value=data.y.values, observed=True)
model = pm.Model([pred, b0, b1, y, sigma, x_obs])
mcmc = pm.MCMC(model)
mcmc.sample(50000, 20000)
print np.mean(mcmc.trace('b1')[:])
print np.mean(mcmc.trace('b0')[:])
print np.mean(mcmc.trace('sigma')[:])
plt.hist(mcmc.trace('b1')[:], bins=50)
plt.show()
