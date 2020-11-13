# Bare-bones GPU Neural Network (from scratch)

## Purpose
A neural network implementation with all the linear algebra to understand the black box. Run on GPU thanks to [Cupy](https://github.com/cupy/cupy)

## Fancy Graphics

### Stochastic Gradient Descent
Note: This is 7840 mini-batches, not epochs (40 epochs were executed)
![Image of Cost Function](/img/L1_7600-->L2_2000-->L3_10_e:7840_a:0.04_acc:100.0%training_performance_with_SGD_98.10%valid.png)
### Validation Accuracy
![Image of Cost Function](/img/L1_7600-->L2_2000-->L3_10_e:40_a:0.04_acc:98.1%validation_performance.png)


## Currently Implements
- Basic Forward Propagation for user-defined network  
- Back Gradient Descent
- Basic Backpropagation and weight updates
- Various Activation functions (Relu, tanh, sigmoid)
- Stochastic Gradient Descent  (mini-batches = *much* faster training)
- L2 Regularization
- Gradient Check for every parameter in model
- Cost function and training accuracy plotting and auto-save  

## Features to Add
- Save/Load trained models
- Drop-Out Regularization
- Momentum Optimization
- RMSprop  
- Adaptive Momentum (Adam) Optimization
- Batch normalization
- Auto hyperparameter tuning

## Resources
### Deep Learning Specialization with Andrew Ng from Coursera
- Week 1 - [Neural Networks and Deep Learning](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
- Week 2 - [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
- Week 3 - [Structuring Machine Learning Projects](https://www.youtube.com/playlist?list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b)
- Week 4 - [Convolutional Neural Networks](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
- Week 5 - [Sequence Models](https://www.youtube.com/playlist?list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6)  
### Michael Nielsen Online Neural Networks Book
- [Using Neural Nets on the MNIST](http://neuralnetworksanddeeplearning.com/chap1.html)
- [How backpropagation works](http://neuralnetworksanddeeplearning.com/chap2.html)
