import numpy as np
import matplotlib.pyplot as plt
from methodSupport import *

x = np.linspace(0,2,3)
print(x)
b = np.random.randn(4)
W = np.random.randn(3,4)
print(b)
print(W)
z = W.T @ x +b
a = 1/(1 + np.exp(-z))
y = np.random.randn(4)
print(a - y)
#print(W.T @ x +b)
x = np.linspace(-5,5,100)
x2 = np.linspace(-1,1,100)
fig,ax = plt.subplots(2,1,figsize=(4,4))
ax[0].plot(x,x**2 + 1)#sigmoid(x),'C3')
ax[1].plot(x2,ReLU(x2),'C2')
ax[1].plot(x2,expReLU(x2,alpha=0.1))
ax[1].plot(x2,LeakyReLU(x2,alpha=0.1))
ax[0].grid(); ax[1].grid()
ax[0].set_title('Sigmoid function')
ax[1].set_title('Rectified Linear Unit')

ax[0].set_xlim([-5.01,5.01]); ax[1].set_xlim([-1.01,1.01])
fig.tight_layout()

plt.show()
















'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Replace y_true and y_pred_prob with your actual data
y_true = np.random.choice([0, 1], size=114, p=[42/114, 72/114])  # replace with actual data
y_pred_prob = np.random.rand(114)  # replace with actual prediction probabilities

# Create a DataFrame for true labels and predicted probabilities
data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})

# Sort by predicted probabilities in descending order
data = data.sort_values(by='y_pred_prob', ascending=False).reset_index(drop=True)

# Calculate cumulative positives and negatives for Class 1 and Class 0
data['cumulative_class_1'] = (data['y_true'] == 1).cumsum()  # cumulative count of true positives (Class 1)
data['cumulative_class_0'] = (data['y_true'] == 0).cumsum()  # cumulative count of true negatives (Class 0)

# Total positives, negatives, and total population
total_class_1 = (y_true == 1).sum()  # 72 in your case
total_class_0 = (y_true == 0).sum()  # 42 in your case
total_population = len(y_true)  # 114 in your case

# Calculate cumulative gain as a percentage of total population for each class
data['cumulative_gain_class_1'] = data['cumulative_class_1'] / total_population * 100
data['cumulative_gain_class_0'] = data['cumulative_class_0'] / total_population * 100

# Calculate population percentage (x-axis)
data['population_percentage'] = np.linspace(0, 100, len(data), endpoint=True)

# Plotting the Cumulative Gains Curve for both Class 0 and Class 1
plt.figure(figsize=(10, 6))

# Cumulative Gains for Class 1 (Positives)
plt.plot(data['population_percentage'], data['cumulative_gain_class_1'], label='Cumulative Gain (Class 1)', color='b', marker='o')

# Cumulative Gains for Class 0 (Negatives)
plt.plot(data['population_percentage'], data['cumulative_gain_class_0'], label='Cumulative Gain (Class 0)', color='r', marker='x')

# Plot baseline (random model)
plt.plot([0, 100], [0, 100], color='gray', linestyle='--', label='Baseline (Random Model)')

# Labeling the plot
plt.xlabel('Percentage of Population')
plt.ylabel('Cumulative Gain (%)')
plt.title('Cumulative Gains Curve for Class 0 and Class 1')
plt.legend()
plt.grid(True)
plt.show()'''






'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])  # actual classes
y_pred_prob = np.array([0.1, 0.9, 0.4, 0.8, 0.7, 0.3, 0.9, 0.2, 0.6, 0.85])  # predicted probabilities

# Create a DataFrame with true labels and predicted probabilities
data = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob})

# Sort the DataFrame by predicted probabilities in descending order
data = data.sort_values(by='y_pred_prob', ascending=False).reset_index(drop=True)

# Calculate cumulative true positives and total positives
data['cumulative_positives'] = data['y_true'].cumsum()
total_positives = data['y_true'].sum()

# Calculate cumulative gain as a percentage
data['cumulative_gain'] = data['cumulative_positives'] / total_positives * 100

# Calculate population percentage
data['population_percentage'] = np.linspace(0, 100, len(data))

# Calculate lift (ratio of cumulative gain to baseline gain)
data['lift'] = data['cumulative_gain'] / data['population_percentage']

# Plotting
plt.figure(figsize=(12, 6))

# Plot Cumulative Gains Curve
plt.plot(data['population_percentage'], data['cumulative_gain'], marker='o', color='b', label='Cumulative Gains')

# Plot Lift Curve
plt.plot(data['population_percentage'], data['lift'], marker='o', color='g', label='Lift Curve')

# Plot baseline
plt.plot([0, 100], [0, 100], color='gray', linestyle='--', label='Baseline')

# Labels and title
plt.xlabel('Percentage of Population')
plt.ylabel('Percentage / Lift')
plt.title('Cumulative Gains and Lift Curve')
plt.legend()
plt.grid(True)
plt.show()
#'''



'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
print(X_train.shape)
print(X_test.shape)
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,y_test)))
#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Logistic Regression
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
#Cross validation
accuracy = cross_validate(logreg,X_test_scaled,y_test,cv=10)['test_score']
print(accuracy)
print("Test set accuracy with Logistic Regression  and scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))


import scikitplot as skplt
y_pred = logreg.predict(X_test_scaled)
print(y_pred)
print(y_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
#plt.show()
y_probas = logreg.predict_proba(X_test_scaled)
print(y_probas.shape)
#skplt.metrics.plot_roc(y_test, y_probas)
#plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()
#'''
'''import numpy as np
# We use the Sigmoid function as activation function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def forwardpropagation(x):
    # weighted sum of inputs to the hidden layer
    z_1 = np.matmul(x, w_1) + b_1
    # activation in the hidden layer
    a_1 = sigmoid(z_1)
    # weighted sum of inputs to the output layer
    z_2 = np.matmul(a_1, w_2) + b_2
    a_2 = z_2
    return a_1, a_2

def backpropagation(x, y):
    a_1, a_2 = forwardpropagation(x)
    # parameter delta for the output layer, note that a_2=z_2 and its derivative wrt z_2 is just 1
    delta_2 = a_2 - y
    print(np.sum(0.5*((a_2-y)**2)))
    # delta for  the hidden layer
    delta_1 = np.matmul(delta_2, w_2.T) * a_1 * (1 - a_1)
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_1.T, delta_2)
    output_bias_gradient = np.sum(delta_2, axis=0)
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(x.T, delta_1)
    hidden_bias_gradient = np.sum(delta_1, axis=0)
    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


# ensure the same random numbers appear every time
np.random.seed(0)
# Input variable
#x = np.array([4.0],dtype=np.float64)
x = np.linspace(0,1,100).reshape(-1,1)
# Target values
y = 2*x+1.0 

# Defining the neural network, only scalars here
n_inputs = x.shape[0]
print(n_inputs)
n_features = 1
n_hidden_neurons = 1
n_outputs = y.shape
print(n_outputs[0])

# Initialize the network
# weights and bias in the hidden layer
w_1 = np.random.randn(n_features, n_hidden_neurons)
print(w_1.shape)
b_1 = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
w_2 = np.random.randn(n_hidden_neurons, n_outputs[0])
b_2 = np.zeros(n_outputs[0]) + 0.01

eta = 0.001
for i in range(500):
    # calculate gradients
    derivW2, derivB2, derivW1, derivB1 = backpropagation(x, y)
    # update weights and biases
    w_2 -= eta * derivW2
    b_2 -= eta * derivB2
    w_1 -= eta * derivW1
    b_1 -= eta * derivB1'''