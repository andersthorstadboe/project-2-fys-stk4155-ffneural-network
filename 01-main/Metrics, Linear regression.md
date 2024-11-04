# Metrics, Linear regression using ADAM, repeated training

layer_output_sizes = [5,5,1]
Regularization, λ: 5e-14
Learning rate,  η: 1e-05
Batches : 32 | epochs : 1000

## Sigmoid
Trained network 4 times
Final best cost: 0.03253924606822291
R²-score       : 0.6292286717317008

## ReLU
Training iterations: 36
Final best cost: 0.007366272903181508
R²-score       : 0.9160643494021633

## LReLU
Trained network 3 times
Final best cost: 0.04118759532264646
R²-score       : 0.5306842883225773

## ELU
Final best cost: 24.884986314693684
R²-score       :-2032.0381342247965

## tanh
Final best cost: 20.3701680889044
R²-score       :-503.96523401372815


# Metrics, Classification using ADAM
layer_output_sizes = [10,5,1]

## sigmoid
(eta,lmbda) 0.001 1e-10
Prediction accuracy, test data    : 0.9305555555555556
Loss: 2.8077299887682554
ROC-AUC: 0.983507
11,01,00,10: 0.94,0.06,0.92,0.08

## ReLU
(eta,lmbda) 0.001 0.001
Prediction accuracy, test data    : 0.9444444444444444
Loss: 0.7903724725175032
ROC-AUC: 0.976562
0.96,0.04,0.92,0.08

## LReLU
(eta,lmbda) 0.001 0.001
Prediction accuracy, test data    : 0.9027777777777778
Loss: 0.7956681528839863
ROC-AUC: 0.955729

## tanh
(eta,lmbda) 0.001 0.001
Prediction accuracy, test data    : 0.9444444444444444
Loss: 1.0639962468024875
ROC-AUC: 0.988715
0.96,0.04,0.92,0.08

(eta,lmbda) 0.001 0.001
Prediction accuracy, test data    : 0.9722222222222222
Prediction accuracy, training data: 0.9818913480885312
Loss: 1.3159399106145824
Loss: 1.3159399106145824
ROC-AUC: 0.993056
0.98 / 0.96

### Current best
#### Linear
ADAM:
Regularization, λ: 5e-14
Learning rate,  η: 1e-05
Training iteration 0
Current cost: 0.08229326614960468
Trained network 3 times
Final best cost: 0.03011046197424892
Final best cost: 0.055625268333052626
R²-score       : 0.6569036677261251
R²-score 0.6569036677261251

#### Classification
ADAM: 
(eta,lmbda) 0.01 1e-09; m = 16; e = 1000
Prediction accuracy, test data    : 0.9583333333333334
Prediction accuracy, training data: 0.9496981891348089
Loss: 5.339760920377584
Loss: 5.339760920377584
ROC-AUC: 0.992188

RMSprop:
(eta,lmbda) 0.004 3e-06; m = 16, e = 1000
Prediction accuracy, test data    : 0.9305555555555556
Prediction accuracy, training data: 0.9617706237424547
Loss: 1.4966939593941775
Loss: 1.4966939593941775
ROC-AUC: 0.973958

(eta,lmbda) 0.003 3e-06, m = 64; e = 1000
Prediction accuracy, test data    : 0.9444444444444444
Prediction accuracy, training data: 0.9798792756539235
Loss: 6.133246919713273
Loss: 6.133246919713273
ROC-AUC: 0.957465

(eta,lmbda) 0.0003 1e-10; m = 128, e = 1000
Prediction accuracy, test data    : 0.9444444444444444
Prediction accuracy, training data: 0.9839034205231388
Loss: 2.711831499427092
Loss: 2.711831499427092
ROC-AUC: 0.993056
0.96 / 0.92

Adagrad
(eta,lmbda) 0.0003 3e-07, m = 64; e = 1000
Prediction accuracy, test data    : 0.9583333333333334
Prediction accuracy, training data: 0.9678068410462777
Loss: 0.7642534029288605
Loss: 0.7642534029288605
ROC-AUC: 0.981771

(eta,lmbda) 0.003 3e-06; m = 64; e = 1000
Prediction accuracy, test data    : 0.9444444444444444
Prediction accuracy, training data: 0.9476861167002012
Loss: 0.8468272423561561
Loss: 0.8468272423561561
ROC-AUC: 0.990451