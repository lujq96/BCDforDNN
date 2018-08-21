# BCD for DNN
This is the code for [Jianqiu Lu](https://github.com/lujq96)'s Undergraduate Thesis. It is used for training DNN by Block Coordinate Descent (Prox-Linear Update) in mini batch.
## Results
We use MNIST dataset.
Accuracy of Block Coordiante Descent training on 1 layer MLP, with structure of 784-800-10, is 
[BCDM-1layerAccu](https://github.com/lujq96/BCDforDNN/blob/master/results/BCDM-1layerAccu.pdf) 
while our baseline, backprop with SGD, has a accuracy of
[sgd_accu](https://github.com/lujq96/BCDforDNN/blob/master/results/sgd_accu.pdf).
Results on 3 layer MLP is
[BCDM-3layerAccu](https://github.com/lujq96/BCDforDNN/blob/master/results/BCDM-3layerAccu.pdf) 
while baseline performance as
[Base_accu](https://github.com/lujq96/BCDforDNN/blob/master/results/fig_accuracy_2.pdf) 