from methodSupport import *
from sklearn.linear_model import Ridge, Lasso
import autograd.numpy as anp
from sklearn.preprocessing import StandardScaler

## --- Gradient descent classes --- ##
class GDTemplate:
    def __init__(self,learning_rate):
        self.eta = learning_rate

    def update_change(self,gradient):
        raise RuntimeError
    
    def reset(self):
        pass

class PlainGD(GDTemplate):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def update_change(self, gradient,theta_m1=None):
        return self.eta * gradient
    
    def reset(self):
        return super().reset()
    
class MomentumGD(GDTemplate):
    def __init__(self, learning_rate,momentum=0.0):
        super().__init__(learning_rate)
        self.mom = momentum
        
    def update_change(self, gradient, theta_m1):
        return self.eta*gradient + self.mom*theta_m1
    
    def reset(self):
        return super().reset()
    
class Adagrad(GDTemplate):
    def __init__(self, learning_rate,momentum=0.0):
        super().__init__(learning_rate)
        self.mom = momentum
        self.adagrad_learn_rate = None

    def update_change(self, gradient,theta_m1):
        delta = 1e-7
        gradient2 = gradient*gradient
        self.adagrad_learn_rate = (self.eta)/(delta + anp.sqrt(gradient2))
        return gradient*self.adagrad_learn_rate + self.mom*theta_m1
    
    def reset(self):
        self.adagrad_learn_rate = None

class RMSprop(GDTemplate):
    def __init__(self, learning_rate,decay):
        super().__init__(learning_rate)
        self.rmsProp_update = 0.0
        self.decay = decay

    def update_change(self, gradient, theta_m1):
        delta = 1e-8
        self.rmsProp_update = self.decay * self.rmsProp_update + (1 - self.decay)*gradient*gradient
        return self.eta/(anp.sqrt(self.rmsProp_update + delta)) * gradient
    
    def reset(self):
        self.rmsProp_update = 0.0
        
class ADAM(GDTemplate):
    def __init__(self, learning_rate, decay_rate_1, decay_rate_2):
        super().__init__(learning_rate)
        self.decay1 = decay_rate_1
        self.decay2 = decay_rate_2
        self.s = 0.0
        self.r = 0.0
        self.t = 1

    def update_change(self, gradient,theta_m1):
        delta = 1e-8
        self.s = self.decay1*self.s + (1. - self.decay1)*gradient
        self.r = self.decay2*self.r + (1. - self.decay2)*gradient*gradient

        s_corr = self.s / (1. - self.decay1**self.t)
        r_corr = self.r / (1. - self.decay2**self.t)

        return self.eta * (s_corr / (anp.sqrt(r_corr) + delta))
    
    def reset(self):
        self.s = 0.; self.r = 0.
        self.t += 1


## --- Helper classes  --- ##
class Initializer:
    def __init__(self,
                 #problem_type: str,
                 problem_case: str='1D',
                 domain: tuple=([0,1],[0,1]),
                 sample_size: list=[10,10],
                 scaling=StandardScaler
                 ):
        
        #self.p_type = problem_type
        self.p_case = problem_case
        self.domain = domain
        self.N      = sample_size
        self.scale  = scaling
        
        self.x, self.y   = None, None
        self.xx, self.yy = None, None
        self.x_noise     = None
        self.y_noise     = None
        self.target      = None

    def domain_setup(self, noise: float=0.):
        """
        
        """
        x0,xN = self.domain[0]; y0,yN = self.domain[1]

        self.x = np.sort(np.random.uniform(x0,xN,self.N[0])).reshape(-1,1)
        self.y = np.sort(np.random.uniform(y0,yN,self.N[1])).reshape(-1,1)

        self.x_noise = noise * np.random.normal(0, noise, self.x.shape) 
        self.y_noise = noise * np.random.normal(0, noise, self.y.shape)

    def test_function(self,coefficients: list):
        """
        
        """

        a = coefficients

        if self.p_case == '1D':
            self.target = [test_func(self.x,a) + self.x_noise, test_func(self.x,a)]
        elif self.p_case == '2D':
            self.xx,self.yy = np.meshgrid(self.x,self.y)
            self.target = exp2D(self.xx,self.yy,a)# + self.x_noise + self.y_noise
        elif self.p_case == 'Franke':
            self.xx,self.yy = np.meshgrid(self.x,self.y)
            self.target = Franke(self.xx,self.yy)# + self.x_noise + self.y_noise

        if self.p_case != '1D':
            self.xf = self.xx.reshape(-1,1); self.yf = self.yy.reshape(-1,1)
            self.target_f = self.target.reshape(-1,1)

    def plot(self, labels=['','','',''], save=False, fig_name='generic name.png'):
        """
        
        """

        if self.p_case == '1D':
            fig = plot1D(self.x,self.target,labels,save,f_name=fig_name)
        else:
            fig = plot2D(self.xx,self.yy,self.target,labels,save,f_name=fig_name)

        return fig

class GradientTemplate:
    def __init__(self,lmbda=0.):
        self.lmbda = lmbda

    def update_gradient(self,X,target,beta):
        raise RuntimeError
    
class OLSGrad(GradientTemplate):
    def __init__(self,
                 lmbda=0
                 ):
        self.lmbda = lmbda

    def update_gradient(self,X,target,beta):
        n_data = len(target)
        return 2 * ((X.T @ (X @ beta - target))/n_data)
    
class RidgeGrad(GradientTemplate):
    def __init__(self, lmbda):
        super().__init__(lmbda)

    def update_gradient(self,X,target,beta):
        return OLSGrad.update_gradient(self,X,target,beta) + 2*self.lmbda*beta 

class LassoGrad(GradientTemplate):
    def __init__(self, lmbda):
        super().__init__(lmbda)

    def update_gradient(self, X, target, beta):
        return OLSGrad.update_gradient(self,X,target,beta) + self.lmbda * anp.sign(beta)
    
class AutoGrad(GradientTemplate):
    def __init__(self,
                 lmbda=0.,
                 cost_function=mse
                 ):
        super().__init__(lmbda)
        self.cost = cost_function

    def update_gradient(self, X, target, beta):
        if self.cost == mse_lasso:
            grad_func = elementwise_grad(self.cost,0)
        else:
            grad_func = grad(self.cost,0)
        return grad_func(beta,X,target,self.lmbda)