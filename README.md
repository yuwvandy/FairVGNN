# FairVGNN
This repository is an official PyTorch(Geometric) implementation of FairVGNN in "Fair View Graph Neural Network for Fair Node Representation Learning". The whole flowchart our model is visualized in the sequal.

![](./img/framework_fairvgnn.png)

## Motivation
Node representation learned by Graph Neural Networks (GNNs) may inherit historical prejudices and societal stereotypes from training data, leading to discriminatory bias in predictions. This paper demonstrates that feature propagation could vary feature correlations and cause the leakage of sensitive information to innocuous feature channels, which could further exacerbate discrimination in predictions.
