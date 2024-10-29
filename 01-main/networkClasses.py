import autograd.numpy as np
from classSupport import *
from methodSupport import *
from typing import Callable
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor


class FFNeuralNework:

    def __init__(self,
                network_input_size,
                layer_output_size,
                activation_functions,
                activation_derivatives,
                cost_function,
                cost_derivative,
                multiple_input=True
        ):
        self.net_in_size = network_input_size
        self.layer_out_size = layer_output_size
        self.multi_input = multiple_input
        self.act_func = activation_functions
        self.act_der = activation_derivatives
        self.cost_func = cost_function
        self.cost_der = cost_derivative

    def create_layers(self):
        """
        Creates the layers based on the initialization of the class
        """
        #print('Create layers:')
        self.layers = []
        i_size = self.net_in_size
        for layer_out_size in self.layer_out_size:
            #if self.multi_input == True:
            W = anp.random.randn(i_size,layer_out_size)
                #print('W_l',W.shape)
            #else:
            #    W = anp.random.randn(layer_out_size,i_size)
            #    print('W_l',W.shape)
            b = anp.random.randn(layer_out_size)
            #print('b_l',b.shape)
            self.layers.append((W,b))

            i_size = layer_out_size

    def feed_forward(self,input):
        #print('Feed-forward')
        self.layer_inputs = []
        self.zs = []
        a = input  # Becomes last layer output, used in back-prop
        for (W, b), a_func in zip(self.layers, self.act_func):
            self.layer_inputs.append(a)
            #if self.multi_input == True:
            '''print('W_l',W.shape)
                print('a_l-1',a.shape)
                print('b_l',b.shape)'''
            z = a @ W + b
                #print('z_l',z.shape)
            '''else:
                print('W_l',W.shape)
                print('a_l-1',a.shape)
                print('b_l',b.shape)
                z = W @ a + b
                print('z_l',z.shape)'''

            
            a = a_func(z)
            #print('a_l',a.shape)
            self.zs.append(z)
        return a

    def back_propagation(self,input,target):
        output_predict = self.feed_forward(input=input)
        self.layers_grad = [() for layer in self.layers]
        #print('Back-prop')
        for i in reversed(range(len(self.layers))):
            #prediction = self.a
            layer_in, z, act_der = self.layer_inputs[i], self.zs[i], self.act_der[i] 
            #if self.multi_input == True:
            if i == len(self.layers)-1:
                    '''print(self.a.shape)
                    print(input.shape)'''

                    dC_da = self.cost_der(output_predict,target)
                    #print('dC_da',dC_da.shape)
            else: 
                    (W,b) = self.layers[i+1]
                    dC_da = dC_dz @ W.T
                    #print('dC_da',dC_da.shape)

                #print('s(z)',act_der(z).shape)
                #print('dC_da',dC_da.shape)
            dC_dz = dC_da * act_der(z)
                #print('dC_dz',dC_dz.shape)
                #print('l_in',layer_in.shape)
            dC_dW = layer_in.T @ dC_dz
                #print('dC_dW',dC_dW.shape)
            dC_db = np.sum(dC_dz,axis=0)
                #print('dC_db',dC_db.shape)
            '''else:
                if i == len(self.layers)-1:
                    print('a_L',output_predict.shape)
                    print('output',input.shape)

                    dC_da = self.cost_der(output_predict,target)
                    print('dC_da_L',dC_da.shape)
                else: 
                    (W,b) = self.layers[i+1]
                    dC_da = dC_dz @ W


                print('s(z)',act_der(z).shape)
                print('dC_da',dC_da.shape)
                dC_dz = dC_da * act_der(z)
                print('dC_dz',dC_dz.shape)
                print('l_in',layer_in.shape)
                dC_dW = np.outer(dC_dz,layer_in)
                print('dC_dW',dC_dW.shape)
                dC_db = dC_dz
                print('dC_db',dC_db.shape)'''

            self.layers_grad[i] = (dC_dW,dC_db)
    
    def train_network(self,input,target,
                      GDMethod: GDTemplate,
                      batches=1,epochs=1000):
        try:
            batch_size = input.shape[0] // batches
        except ZeroDivisionError:
            batch_size = 1
        for e in range(epochs):
            GDMethod.reset()
            for _ in range(batches):
                rand_idx = batch_size*anp.random.randint(batches)
                input_i,target_i = input[rand_idx:rand_idx+batch_size], target[rand_idx:rand_idx+batch_size]

                self.back_propagation(input_i,target_i)
                
                for (W,b),(dW,db) in zip(self.layers,self.layers_grad):
                    W -= GDMethod.update_change(dW,W)
                    b -= GDMethod.update_change(db,b)

    def score(self):
        raise NotImplementedError
        
    
class LinearRegressor:
    def __init__(self,
                 cost_function=mse,
                 learning_rate=0.1,
                 num_iterations=1,
                ):
        
        self.cost_func = cost_function
        self.eta = learning_rate
        self.n_iter = num_iterations

        self.beta_gd = None
        self.beta_linreg = None

    def design_matrix(self, x, poly_deg=1, intercept=False):
        self.intercept = intercept
        if len(x) == 2:
            X = poly_model_2d(x[0],x[1],poly_deg,intercept)
        else:
            X = poly_model_1d(x,poly_deg,intercept)
        return X

    def scale(self,data,scaler=StandardScaler()):
        scaler.fit(data)
        data_s = scaler.transform(data)

        return data_s

    def reg_fit(self, X, target,
                GDMethod: GDTemplate, GradMethod: GradientTemplate,
                batches=1,epoch=100):
        
        n_data,n_features = X.shape
        self.beta_gd = anp.random.randn(n_features,1)
        #self.beta_gd = anp.zeros((n_features,1))
        try:
            batch_size = X.shape[0] // batches
        except ZeroDivisionError:
            batch_size = 1

        if batch_size <= 1:
            for _ in range(self.n_iter):
                grad = GradMethod.update_gradient(beta=self.beta_gd,X=X,target=target)
                self.beta_gd -= GDMethod.update_change(grad,self.beta_gd)
                #print(self.beta_reg)
        else:
            for e in range(epoch):
                GDMethod.reset()
                for _ in range(batches):
                    rand_idx = batch_size*anp.random.randint(batches)
                    Xi,target_i = X[rand_idx:rand_idx+batch_size], target[rand_idx:rand_idx+batch_size]

                    grad = GradMethod.update_gradient(beta=self.beta_gd,X=Xi,target=target_i)
                    self.beta_gd -= GDMethod.update_change(grad,self.beta_gd)

        target_mean = np.mean(target); X_mean = np.mean(X,axis=0)
        self.gd_intcept = np.mean(target_mean - X_mean @ self.beta_gd)


    def linear_predict(self, X, target, predict_type='OLS',lmbda=0.):
        I = anp.eye(X.shape[1],X.shape[1])
        if predict_type == 'OLS':
            try:
                self.beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ target
            except np.linalg.LinAlgError:
                self.beta_linreg = SVDcalc(X) @ target
        elif predict_type == 'Ridge':
            self.beta_linreg = np.linalg.inv(X.T @ X - lmbda * I) @ X.T @ target
        elif predict_type == 'Lasso':
            reg_lasso = Lasso(lmbda,fit_intercept=False,max_iter=5000)
            reg_lasso.fit(X,target)
            self.beta_linreg = reg_lasso.coef_
        else:
            raise TypeError('Invalid input in "predict_type". Valid: "OLS","Ridge","Lasso"')
        
        target_mean = np.mean(target); X_mean = np.mean(X,axis=0)
        self.intcept = np.mean(target_mean - X_mean @ self.beta_linreg)

    def predict(self,beta,X,target):
        return self.cost_func(beta,X,target)
    
    def plot(self,x_data, y_data,
             labels=['','','','',''], save=False, fig_name='generic name.png'
            ):
        
        if len(x_data) == 2:
            plot2D(x_data[0],x_data[1],y_data,labels,save,fig_name)
        
        else:
            plot1D(x_data,y_data,labels,save,fig_name)

        #return fig

    def reset(self):
        """
        Resetting some instance variables to be able to reuse class in loops
        """
        self.beta_gd = None
        self.gd_intcept = None


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
    
    def gd_fit(self,inputs,target, GDMethod: GDTemplate, lmbda=0. ,batches=1, epoch=100):
        n_data,n_features = inputs.shape
        self.beta_lg = anp.random.randn(n_features)
        try:
            batch_size = inputs.shape[0] // batches
        except ZeroDivisionError('Num. of batches causes division by zero'):
            batch_size = 1

        if batches <= 1:
            for _ in range(self.n_iter):
                y_predict = self.cost_func((inputs @ self.beta_lg))
                grad = 2 * ((inputs.T @ (y_predict - target))/n_data + lmbda*self.beta_lg)
              
                self.beta_lg -= GDMethod.update_change(grad,self.beta_lg)

        else:
            for e in range(epoch):
                for _ in range(batches):
                    rand_idx = batch_size*anp.random.randint(batches)
                    Xi, yi = inputs[rand_idx:rand_idx+batch_size], target[rand_idx:rand_idx+batch_size]

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

class ScikitLearnRegressor:
    def __init__(self,
                 cost_function=mse):
        self.cost = cost_function

    def gd_fit(self):
        sdg_reg = SGDRegressor()

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

    X = lin_model.design_matrix(x,poly_deg=2)

    lin_model.reg_fit(X,y,GDMethod,0.01,batches,epoch=1000)
    beta_lin = lin_model.linear_predict(X,y,'Ridge',0.01)
    #model.gd_fit(X,y,GDMethod,0.01,batches)
    print(lin_model.predict(X @ lin_model.beta_linreg,y))
    print(lin_model.predict(X @ lin_model.beta_gd,y))
    #print(model.predict(X,binary=True))


