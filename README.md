# BCD for DNN
This is the code for [Jianqiu Lu](https://github.com/lujq96)'s Undergraduate Thesis. It is used for training DNN by Block Coordinate Descent (Prox-Linear Update) in mini batch.
## Results
We use MNIST dataset.
Accuracy of Block Coordiante Descent training on 1 layer MLP, with structure of 784-800-10, is 
![Jianqiu LU](https://raw.githubusercontent.com/lujq96/BCDforDNN/4d668195cfc8ebd65db322c58d24f9bd542a0a8f/results/BCDM-1layerAccu.pdf) 
while our baseline, backprop with SGD, has a accuracy of
![Jianqiu LU](https://raw.githubusercontent.com/lujq96/BCDforDNN/4d668195cfc8ebd65db322c58d24f9bd542a0a8f/results/sgd_accu.pdf).
Results on 3 layer MLP is
![Jianqiu LU](https://raw.githubusercontent.com/lujq96/BCDforDNN/4d668195cfc8ebd65db322c58d24f9bd542a0a8f/results/BCDM-3layerAccu.pdf) 
while baseline performance as
![Jianqiu LU](https://raw.githubusercontent.com/lujq96/BCDforDNN/4d668195cfc8ebd65db322c58d24f9bd542a0a8f/results/fig_accuracy_2.pdf) 