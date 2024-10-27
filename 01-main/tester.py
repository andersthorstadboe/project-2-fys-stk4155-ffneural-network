#from sys import path
#path.append('project-02-fys-stk4155/01-main/')

from networkClasses import *
from classSupport import *
from methodSupport import *

import autograd.numpy as anp
import numpy as np
from autograd import grad, elementwise_grad
import jax.numpy as jnp
import jax

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

cases = ['1D','2D','Franke']
case_ = cases[2]
show = True

# Grid and data setup
a      = [1.0, 1.5, 1.2]                     # Coefficients for exponential model
c0, c1 = 0.1, 0.95                           # Noise scaling    
x0, xN = 0, 1                                # Start and end of domain, x-axis
y0, yN = 0, 1                                # Start and end of domain, y-axis
Nx, Ny = 5, 5                             # Number of sample points

dataset = Initializer(problem_case=case_,sample_size=[Nx,Ny])
dataset.domain_setup()#noise=0)
dataset.test_function(a)
f = dataset.plot(labels=['Test function','x','y','dataset','true'])
#plt.show()

lin_gd = LinearRegressor()

X = lin_gd.design_matrix(dataset.x,poly_deg=2)

scale = StandardScaler()

scale.fit(X)
X_s = scale.transform(X)

print(scale.mean_)
print(X)
print(X_s)
