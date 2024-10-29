import autograd.numpy as anp
import numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns

## --- Activation functions --- ##
def identity(z):
   return z

def sigmoid(z):
    return 1 / (1 + anp.exp(-z))
   
def softmax(z):
   """
   Computes softmax values for each set of scores in the rows of the matrix z. 
   """
   e_z = anp.exp(z - anp.max(z,axis=0))
   return e_z / anp.sum(e_z,axis=1)[:,anp.newaxis]
   
def softmax_vec(z):
   e_z = anp.exp(z - anp.max(z))
   return e_z / anp.sum(e_z)
    
def ReLU(z):
    return anp.where(z > 0, z, 0)

def ReLU_der(z):
   return anp.where(z > 0, 1, 0)

def LeakyReLU(z,alpha=0.01):
   return anp.where(z >= 0, z, alpha*z)
    
def expReLU(z,alpha=1.):
    return anp.where(z >= 0, z, alpha*(anp.exp(z)-1))
        #return [alpha*(anp.exp(z)-1) if val < 0 else z for val in z]

    
## --- Cost functions --- ##
def mse(beta,X,target,lmbda=None):
   return anp.sum((X @ beta - target)**2)/(target.shape[0])

def mse_ridge(beta,X,target,lmbda):
   return mse(beta,X,target) + lmbda*anp.sum(beta**2)

def mse_lasso(beta,X,target,lmbda):
   return mse(beta,X,target) + lmbda*anp.abs(beta)

def mse_predict(prediction,target):
   return anp.sum((prediction - target)**2)/(target.shape[0])

def mse_predict_l2(prediction,target,lmbda):
   return mse_predict(prediction,target) + lmbda*anp.sum()
    
def cross_entropy(predict,target):
   return anp.sum(-target * anp.log(predict))

## --- Support functions --- ##
def poly_model_1d(x: np.ndarray, poly_deg: int, intcept=False):
   """
   Returning a design matrix for a polynomial of a given degree in one variable, x. 
   Includes the β0-column in building X, but this is taken out from the output by default.

   Parameters
   -------
   x : NDArray
      Dimension n x 1
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
   intcept : bool
      If True, X is return with the β0-column
    
   Returns
   --------
   numpy.ndarray : Design matrix, X. Dimension: n x p or n x (p-1)
   """

   X = np.zeros((len(x),poly_deg+1))
   for p_d in range(poly_deg+1):
      X[:,p_d] = x[:,0]**p_d

   if intcept == True:
      return X
   else:
      return X[:,1:]

def poly_model_2d(x: np.ndarray, y: np.ndarray, poly_deg: int, intcept=False):
   """ From lecture notes
   Returning a design matrix for a polynomial of a given degree in two variables, x, y.
   Includes the β0-column in building X, but this is taken out from the output by default.
   
   Parameters
   ---
   x, y : NDArray
      Mesh in x and y direction
   poly_deg : int
      degree of the resulting polynomial to be modelled, p
   intcept : bool
      If True, X is return with the β0-column
   
   Returns
   ---
   numpy.ndarray : Design matrix, X. Dimension: n x (0.5*(p+2)(p+1))-1
   """

   if len(x.shape) > 1:
      x = np.ravel(x); y = np.ravel(y)

   cols = int(((poly_deg + 2) * (poly_deg + 1))/2)
   X = np.ones((len(x),cols))

   for p_dx in range(poly_deg+1):
      q = int((p_dx+1)*(p_dx)/2)
      for p_dy in range((p_dx+1)):
         #print('q, p_dx, p_dy = ',q, p_dx, p_dy)
         #print('x^%g * y^%g' %((p_dx-p_dy),p_dy))
         X[:,q+p_dy] = (x**(p_dx-p_dy)) * (y**p_dy)

   if intcept == True:
      return X
   else:
      return X[:,1:]
   
def SVDcalc(X):
   """
   Calculating the (X^T X)^-1 X^T - matrix product using a singular value decomposition, SVD
   of X, as X = U S V^T

   Parameters
   ---
   X : ndarray, n x p

   Returns
   ---
   NDArray : matrix product V (S^T S)^-1 S^T U^T 
   """
   U,S_tmp,V_T = np.linalg.svd(X)
   S = np.zeros((len(U),len(V_T)))
   S[:len(X[0,:]),:len(X[0,:])] = np.diag(S_tmp)
   STS = S.T @ S

   return V_T.T @ np.linalg.pinv(STS) @ S.T @ U.T

## --- Test functions --- ##
def test_func(x: np.ndarray, a: list=None):
   """
   Parameters
   ---
   x : ndarray, n
      Data points
   a : list
      List of coefficients (floats)

   Returns
   ---
   ndarray : 1D polynomial function: a[i]x**i, for i = 0,1,...,len(a)
   """
   f = anp.zeros(x.shape)
   for i in range(len(a)):
      f += a[i]*x**i
   return f
    
def Franke(x,y):
   """
   Returns the Franke function on a mesh grid (x,y) as data function, with or without noise added
   
   Parameters
   --------
   x, y : array, n x m
      Mesh grid data points

   Returns
   --------
   ndarray : 2D-function representing the Franke function

   """
   p1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
   p2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
   p3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
   p4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

   return p1 + p2 + p3 + p4

def exp1D(x: np.ndarray, a: list):
   """
   Returns a 1D exponential test function, "a*exp(-x²) + b*exp(-(x-2)²) + noise" for a ndarray x, with or without noise given as ndarray x_noise

   Parameters
   --------
   x : ndarray, n
      Data points
   x_noise : ndarray, n
      Noise data array
   a : list
      List of coefficients (floats)

   Returns
   --------
   ndarray : 1D exponential function
   """
   return a[0]*np.exp(-x**2) + a[1]*np.exp(-(x-2)**2)

def exp2D(x,y,a):
   """
   Returns a 2D exponential test function, for a mesh grid (x,y) with or without noise from mesh grid (x_noise, y_noise)

   Parameters
   ---
   x, y : array, n x m
      Mesh grid data points
   a : list
      List of coefficients (floats)
   
   Returns
   ---
   ndarray : 2D exponential function
   """
   return a[0]*np.exp(-(x**2 + y**2)) + a[1]*np.exp(-(x-2)**2 - (y-2)**2)

## --- Plotting functions --- #
def plot1D(x_data, z_data, labels=['','','','','',''],
           save=False, f_name: str='generic name.png'):
   """
   Returns a surface plot of 2D-data functions

   Parameters
   ---
   x_data : NDArray
      np.ndarray of x-axis data
   z_data : NDArray
      np.ndarray to be plotted against x_data
   labels : list
      List of figure labels. 0: axes-title; 1,2: x-,y-axis labels; 3: scatter-plot label; 4: line-plot label
   save : bool
      Default = False. Saves figure to current folder if True
   f_name : str
      file-name, including file-extension

   Returns
   ---
   Figure is initialized, must be shown explicitly

   """
   if save == True:
      plt.rcParams["font.size"] = 10
      fig,ax = plt.subplots(1,1,figsize=(3.5,(5*3/4)))
   else:
      fig,ax = plt.subplots(1,1)

   line_styles = [None,'--','-.']
   if type(z_data) == list:
      if len(labels) < 3 + len(z_data):
         for i in range(len(z_data)-2):
            labels.append('')
         print('Not enough labels, list extended with empty instances as:')
         print('labels =',labels)
   
   # Plotting initial data
   if type(z_data) != list:
      ax.plot(x_data,z_data,label=labels[3])
   else:
      ax.scatter(x_data,z_data[0],label=labels[3],color='0.15',alpha=0.65)
      for i in range(1,len(z_data)):
         ax.plot(x_data,z_data[i],label=labels[4+(i-1)],ls=line_styles[i-1])

   ax.set_title(labels[0]) 
   ax.set_xlabel(labels[1]); ax.set_ylabel(labels[2],rotation=0,labelpad=10)
   ax.legend(); ax.grid()
   if save == True:
      fig.savefig(f_name,dpi=300,bbox_inches='tight')

   return fig,ax

from mpl_toolkits.mplot3d import Axes3D
def plot2D(x_data, y_data, z_data, labels=['','','','','',''],
           save=False, f_name: str='generic name.png'
           ):
   """
   Returns a surface plot of 2D-data functions

   Parameters
   ---
   x_data : NDArray
      np.ndarray of x-axis data created with np.meshgrid(x,y)
   y_data : NDArray
      np.ndarray of y-axis data created with np.meshgrid(x,y)
   z_data : NDArray
      np.ndarray to be plotted on (x_data,y_data)-gird
   labels : list
      List of figure labels. 0: axes-title; 1,2,3: x-,y-, z-axis labels
   save : bool
      Default = False. Saves figure to current folder if True
   f_name : str
      file-name, including file-extension

   Returns
   ---
   Figure is initialized, must be shown explicitly

   """
   if save == True:
      plt.rcParams["font.size"] = 10
      fig = plt.figure(figsize=(4.5,(5*3/4)))
   else:
      fig = plt.figure()#figsize=(6,3))
   # Plotting initial data
   ax = fig.add_subplot(111,projection='3d')
   f1 = ax.plot_surface(x_data,y_data,z_data,cmap='viridis')
   ax.set_aspect(aspect='auto')
   #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, 0.5, 1]))  # Scale Y more than X
   #fig.subplots_adjust(left=0, right=10, top=1, bottom=0)
   ax.view_init(elev=25, azim=-30)
   ax.set_title(labels[0]); ax.set_xlabel(labels[1])
   ax.set_ylabel(labels[2]); ax.set_zlabel(labels[3])
   ax.tick_params(axis='both', which='major', labelsize=6)
   #fig.tight_layout()
   if save == True:
      fig.savefig(f_name,dpi=300,bbox_inches='tight')

   return fig,ax

import matplotlib.colors as mcolors

def lambda_eta(data, axis_vals, axis_tick_labels=['',''],
               cbar_lim=[-10,10], cmap='viridis', 
               save=False, f_name='generic name.png'
               ):
   """
   Plotting a heatmap of input data using the Seaborn heatmap-method. 
   Default setup with axis-labels for comparing regression parameter λ- and learning rate, η.
   Plot can be modified by using the outputed fig,ax-objects with standard Matplotlib.pyplot-commands
   """
   if save == True:
      plt.rcParams["font.size"] = 10
      fig,ax = plt.subplots(1,1,figsize=(3.5,(5*3/4)))
   else:
      fig,ax = plt.subplots(1,1)
   
   mask = np.isnan(data) #(data==1)
   #cmp = sns.color_palette(cmap, as_cmap=True)
   #cmp = mcolors.ListedColormap(['gray'] + list(cmp(np.linspace(0, 1, 256))))

   if len(axis_tick_labels) < 2: # Condition since we need tick-types for both axis, and only one is provided
      print('ListLengthWarning: Only one tick-label provided, using the same for 2nd axis')
      axis_tick_labels.append(axis_tick_labels)

   ax = sns.heatmap(data=data,vmin=cbar_lim[0],vmax=cbar_lim[1],cmap=cmap,annot=True,mask=mask)
   
   for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if mask[i, j]:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color='0.15', edgecolor='none'))
            ax.text(j + 0.5, i + 0.5, f"{data[i, j]:.1f}", ha='center', va='center', color="white")

   ax.set_xticks(np.arange(len(axis_vals[0])),labels=axis_tick_labels[0])
   ax.set_yticks(np.arange(len(axis_vals[1])),labels=axis_tick_labels[1],rotation=0)
   ax.set_xlabel('λ'); ax.set_ylabel('η',rotation=0,labelpad=10)

   if save == True:
      fig.savefig(f_name,dpi=300,bbox_inches='tight')
           
   return fig,ax