# FYS-STK4155 - Project 2 - Feed-Forward Neural Network for linear regression and classification
Repository for Project 2 in FYS-STK4155 - FF Neural Network for linear and logistic regression<br /><br />
Here you'll find the programs, project report, and some additional results produced for the second project in the subject FYS-STK4155 at UiO.<br /><br />
The folder structure is:
- **01-main**: <br />Folder for the notebooks, programs, class-files and other support files used in the project.
- **02-report**: <br /> The written report and .tex-files used to generate the report. The report abstract, introduction and some conclusions are given below.
- **04-figures**: <br />A selection of figures and results generated during the project. A file named _Metrics.md_ shows some parameter configurations that produced good results.
- **09-archive**: <br />Preliminary files not used in the final version of the project
<br /><br />

## Running the project programs
The project programs can be found in the **01-main** folder. They are organized in different Jupyter notebooks, and can be run given that the user downloads all files in **01-main**, and keeps them in a shared directory. <br />

The notebooks imports the classes and methods from three separate files. The main class file is _networkClasses.py_, which depends on methods from both _classSupport.py_ and _methodSupport.py_. <br />

The notebook also depends on a number of different python packages, such as _autograd_, _scikit-learn_, _seaborn_, _matplotlib_ and _numpy_, so these are required to run the notebook.

## Project abstract and introduction, and main conclusions and findings
### Abstract
_In this project, I am implementing three different algorithms with the aim of studying their performance against two different types of datasets, namely a continuous 2D-dataset for regression, and a binary classification dataset. The algorithms are one general purpose Feed-Forward Neural Network (FFNN), two Stochastic Gradient Descent (SGD)-algorithms, one for linear regression, and one for logistic regression. The FFNN is the main focus of the study, and I am using the two others, in addition similar Scikit-learn methods and standard linear regression, to validate the performance of the FFNN-algorithm. My study uses five different SGD-methods, and five activation functions to investigate how this choice impacts the behavior and outcomes for the network. The analysis will also address the effect from different hyperparameters such as learning rate, regularization and mini-batches, performing grid-searches to locate optimal settings for these. For testing I'm using datasets generated from the 1st Franke function for the regression analysis, and the Wisconsin Breast Cancer Data for the classification analysis. I found the best performance for the regression case with the ReLU-activation function in the hidden layers, but the network struggled with this dataset. For the classification case, the network performed well, producing predictions that scored similarly to other model._
### Introduction
When dealing with real world data, there is a lot of inherent uncertainty in the datasets. Developing general purpose methods and algorithms for dealing with this uncertainty for predictions and approximations based on the data is a major driver behind the development of the class of methods named neural networks. Having general-purpose templates to produce specialized models have introduced a flexible approach to solving these real-world challenges.<br />

The aim of this project is to show how to use a general _Feed-Forward Neural Network_ (FFNN) to make predictions on two types of data. These types are a 2D-dataset generated for the first Franke function, and a classification dataset generated for the Wisconsin Breast Cancer Data.<br />

I will be presenting how the network behavior changes with different choices of parameters and method. Examples of these parameters are learning rate, regularization, and mini-batch sizes. The methods are different _stochastic gradient descent_ (SGD)-methods, such as _momentum gradient descent_ and _ADAM_. I will also see how the choices of different so-called activation functions for the layers in the FFNN impact the performance of the network.<br />

As part of the project, I also implemented two SGD-regression algorithms, one for linear regression, and one for logistic regression, comparing the to the FFNN-implementation, and also to investigate their applications. These two methods also use the same SGD-methods for their gradient calculations.<br />

For the purposes of verification and validation, I also use \textbf{scikit-learn}'s MLPRegressor and MLPClassifier-classes, treating them as benchmark implementations to test against \cite{scikit-learn}. In addition, I will also be using the my own results from a linear regression study looking at the Franke-function for validation of the linear regression. That report can be found here: {https://github.com/andersthorstadboe/project-1-fys-stk4155-lin-regression/tree/main/02-report}.<br />

The report will, in Sec.II, go through the theoretical concepts relevant for the FFNN-algorithm, explaining the general building blocks of the network, the gradient descent methods, the backpropagation algorithm, and different activation functions for the network layers. This part will also address model selection with regards to parameter selection, and model assessment using different measures. Examples of these are mean squared error, R$^{2}$-score, model accuracy, and the confusion matrix.<br />

The Sec.III gives an overview of how algorithms for the FFNN, and the two SGD-methods, are implemented. It also presents the general structure of the programs and a short introduction on how to access and use the different programs used in this analysis.<br />

The final parts, Secs.IV,V of the report addresses a selection of results from the analysis of the two datasets, and presents some of main findings. Some notable results are that the network struggles with the regression dataset, but performs a lot better with the classification case. The best performance for the two datasets are MSE $\approx 0.007$, R$^{2} \approx 0.92$ for the regression analysis, and an accuracy of $\approx 0.96$ for the classification dataset. The report also has an appendix that presents some of the theoretical concepts in more detail, and also some additional results that may be of the interest to the reader. 

### Conclusion and findings
Here are some of the main findings from the project:
1. The network struggles with the regression analysis,especially when the complexity of the target function increases.
2. It performs better on the classification task, reaching accuracies and ROC-AUC scores similar to the methods used for comparison
3. Making the networks deeper or more complex (with more nodes in the layers) made the model performance worse
4. The SGD-methods both for classification and linear regression performed well compared to the other methods.


