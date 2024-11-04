from networkClasses import *
from classSupport import *
from methodSupport import *

import autograd.numpy as anp
from autograd import grad,elementwise_grad
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

## Random seed
def_seed = 1
np.random.seed(def_seed); anp.random.seed(def_seed)

## Figure defaults
plt.rcParams["figure.figsize"] = (8,3); plt.rcParams["font.size"] = 10

anp.random.seed(def_seed)
cases = ['1D','2D','Franke']
case_ = cases[2]
show = True #True False

# Grid and data setup
a      = [1.0, 1.5, 1.2]                                   # Coefficients for exponential model
c0, c1 = 0.2, 0.95                                         # Noise scaling    
x0, xN = 0, 1                                              # Start and end of domain, x-axis
y0, yN = 0, 1                                              # Start and end of domain, y-axis
Nx, Ny = 50, 50                                            # Number of sample points

dataset = Initializer(problem_case=case_,sample_size=[Nx,Ny])
dataset.domain_setup(noise=c0)
dataset.test_function(a)
if case_ == '1D':
    f = dataset.plot(labels=['Test function','x','y','dataset','true'])
else:
    f = dataset.plot(labels=['Test function','X','Y','Z'])

if case_ == '1D':
    targets = dataset.target[0]
    inputs = dataset.x

else:
    targets = dataset.target_f
    x = dataset.xf; y = dataset.yf
    inputs = anp.zeros((x.shape[0],2))
    inputs[:,0] = x[:,0]
    inputs[:,1] = y[:,0] 

test_size = 1/5
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,targets,test_size=test_size,random_state=def_seed)
#inputs_train, inputs_test, targets_train, targets_test = inputs,inputs,targets,targets
## Data scaling
scaler = StandardScaler() #StandardScaler() MinMaxScaler()
scaler.fit(inputs_train)
scale = False
if scale == True:
    inputs_train_s = scaler.transform(inputs_train)
    inputs_test_s = scaler.transform(inputs_test)
else:
    inputs_train_s = inputs_train
    inputs_test_s = inputs_test

n_inputs,n_features = inputs_train_s.shape

layer_output_sizes = [5,10,1]

hidden_func  = sigmoid #sigmoid ReLU, expReLU, LeakyReLU,identity
hidden_der = sigmoid_der #elementwise_grad(hidden_func,0)

act_funcs = []; act_ders = []
for i in range(len(layer_output_sizes)-1):
    act_funcs.append(hidden_func)
    act_ders.append(hidden_der)
act_funcs.append(identity); 
output_der = identity #elementwise_grad(act_funcs[-1],0);
act_ders.append(output_der)

cost_func = mse_predict
cost_der  = grad(cost_func,0)

network = FFNNetwork(network_input_size=n_features,layer_output_size=layer_output_sizes,
                         activation_functions=act_funcs,activation_derivatives=act_ders,
                         cost_function=cost_func,cost_derivative=cost_der)
network.reset()
network.create_layers()

## Gradient Descent setup
eta = 1e-5
gamma = 0.00000001
lmbda = 0.0001; lp = 2
batches = 32; epoch = 1000
decay_rms = 0.9
adagrad_mom = 0.00000000000000000001
ADAM_decay = [0.9, 0.99]

## Calling the gradient descent (GD)-method
#GDMethod = [PlainGD(eta,lmbda=lmbda,lp=lp),PlainGD(eta,lmbda=lmbda,lp=lp)]
#GDMethod = [MomentumGD(eta,gamma,lmbda=lmbda,lp=lp),MomentumGD(eta,gamma,lmbda=lmbda,lp=lp)]
#GDMethod = Adagrad(eta,adagrad_mom,lmbda=lmbda,lp=lp)
#GDMethod = [RMSprop(eta,decay=decay_rms),RMSprop(eta,decay_rms)] 
GDMethod = ADAM(eta,ADAM_decay)

for i in range(2):

    network.train_network(inputs_train_s,targets_train,GDMethod,batches=batches,epochs=epoch)


final_predict = network.feed_forward(inputs_test_s)
show = True
if show == True:
    print(f'Method: {GDMethod.__class__.__name__}')
    print('Regularization, λ:',lmbda)
    print('Learning rate,  η:',eta)
    print('MSE: ',mse_predict(final_predict,targets_test))

final_full_fit = network.feed_forward(inputs)
if case_ == '1D':
    plot1D(input,[targets_test,final_predict],
           labels=[f'Network prediction\nGD-method: {GDMethod.__class__.__name__}, {hidden_func.__class__.__name__}'
                                                ,'x','y','f (x)','ỹ (x)','',''])
else:
    final_fit = final_full_fit.reshape(Nx,Ny)
    plot2D(dataset.xx,dataset.yy,final_fit,
           labels=[f'Network prediction\nGD-method: {GDMethod.__class__.__name__}'
                                                   ,'X','Y','Z'])
    

plt.show()