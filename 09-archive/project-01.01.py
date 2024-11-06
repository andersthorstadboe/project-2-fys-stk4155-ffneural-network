import autograd.numpy as np
from networkClasses import *
from classSupport import *
from methodSupport import *
from typing import Callable


class FFNeuralNework:

    def __init__(self,
                network_input_size,
                layer_output_size
        ):
        self.net_in_size = network_input_size
        self.layer_out_size = layer_output_size

    def create_layers(self):
        """
        Creates the layers based on the initialization of the class

        """
        self.layers = []
        i_size = self.net_in_size
        for layer_out_size in self.layer_output_sizes:
            W = np.random.randn(i_size,layer_out_size)
            b = np.random.randn(layer_out_size)
            self.layers.append((W,b))

            i_size = layer_out_size

    
class LinearRegressor:

    def __init__(self,
                 cost_function=mse,
                 learning_rate=0.1,
                 num_iterations=1
                ):
        self.cost_func = cost_function
        self.eta = learning_rate
        self.n_iter = num_iterations
        self.beta_reg = None

    def reg_fit(self,X,y, GDMethod: GDTemplate, lmbda=0.,batches=1,epoch=100):
        n_data,n_features = X.shape
        self.beta_reg = anp.random.randn(n_features,1)
        try:
            batch_size = X.shape[0] // batches
        except ZeroDivisionError:
            batch_size = 1
        if batch_size <= 1:
            for _ in range(self.n_iter):
                
                grad = 2 * ((X.T @ (X @ self.beta_reg - y))/n_data + lmbda*self.beta_reg)
                self.beta_reg -= GDMethod.update_change(grad,self.beta_reg)
        else:
            for e in range(epoch):
                for _ in range(batches):
                    rand_idx = batch_size*anp.random.randint(batches)
                    Xi,yi = X[rand_idx:rand_idx+batch_size], y[rand_idx:rand_idx+batch_size]

                    grad = 2 * ((Xi.T @ (Xi @ self.beta_reg - yi))/n_data + lmbda*self.beta_reg)
                    self.beta_reg -= GDMethod.update_change(grad,self.beta_reg)


    def predict(self,X,target):
        return self.cost_func(X @ self.beta_reg,target)


class LogisticRegressor:
    def __init__(self,
                 cost_function=sigmoid,
                 learning_rate=0.1,
                 num_iterations=1
                ):
        self.cost_func = cost_function
        self.eta = learning_rate
        self.n_iter = num_iterations
        self.beta_lg = None
    
    def gd_fit(self,X,y, GDMethod: GDTemplate, lmbda=0.,batches=1,epoch=100):
        n_data,n_features = X.shape
        self.beta_lg = anp.random.randn(n_features)
        try:
            batch_size = X.shape[0] // batches
        except ZeroDivisionError:
            batch_size = 1

        if batches <= 1:
            for _ in range(self.n_iter):
                y_predict = self.cost_func((X @ self.beta_lg))
                grad = 2 * ((X.T @ (y_predict - y))/n_data + lmbda*self.beta_lg)
              
                self.beta_lg -= GDMethod.update_change(grad,self.beta_lg)

        else:
            for e in range(epoch):
                for _ in range(batches):
                    rand_idx = batch_size*anp.random.randint(batches)
                    Xi,yi = X[rand_idx:rand_idx+batch_size], y[rand_idx:rand_idx+batch_size]

                    y_predict = self.cost_func((Xi @ self.beta_lg))

                    #print(Xi.shape,self.beta_lg.shape)
                    #print(y_predict.shape)
                    grad = 2 * (Xi.T @ (y_predict - yi)/n_data + lmbda*self.beta_lg)
              
                    self.beta_lg -= GDMethod.update_change(grad,self.beta_lg)

    
    def predict(self,X,binary=False):
        y_pred = self.cost_func(X @ self.beta_lg)
        if binary == True:
            return [1 if i >= 0.5 else 0 for i in (y_pred)]
        else:
            return y_pred
        
if __name__ == '__main__':
    anp.random.seed(1)
    X = anp.array([[0, 0], [1, 0], [0, 1], [1, 1]])#, [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]])
    y = anp.array([0, 0, 0, 1])#, 0, 0, 0, 1, 0, 0, 0, 1])  # This is an AND gate

    n = 100; x0,xN = 0,1
    a = np.random.rand(3); d = 0.0

    x = np.linspace(x0,xN,n).reshape(-1,1) #2*np.random.rand(n,1).reshape(-1,1)
    y = test_func(x,a) + np.random.normal(0, d, x.shape)

    X = poly_model_1d(x,3)

    eta = 0.1; n_iter = 1000; decay = 0.9; decay2 = 0.99
    batch_size = 1
    batches = int(X.shape[0]/batch_size)
    lin_model = LinearRegressor(learning_rate=eta,num_iterations=n_iter)
    model = LogisticRegressor(learning_rate=eta,num_iterations=n_iter)
    
    #GDMethod = PlainGD(eta)
    #GDMethod = MomentumGD(eta,momentum=0.001)
    #GDMethod = Adagrad(eta,momentum=0.001)
    GDMethod = RMSprop(eta,momentum=0.001,decay=decay)
    #GDMethod = ADAM(eta,decay_rate_1=decay,decay_rate_2=decay2)

    lin_model.reg_fit(X,y,GDMethod,0.01,batches,epoch=1000)
    #model.gd_fit(X,y,GDMethod,0.01,batches)
    
    print(lin_model.predict(X,y))
    #print(model.predict(X,binary=True))


