# FairVGNN
This repository is an official PyTorch(Geometric) implementation of FairVGNN in "Fair View Graph Neural Network for Fair Node Representation Learning".

## Motivation
Node representation learned by Graph Neural Networks (GNNs) may inherit historical prejudices and societal stereotypes from training data, leading to discriminatory bias in predictions. This paper demonstrates that feature propagation could vary feature correlations and cause the leakage of sensitive information to innocuous feature channels, which could further exacerbate discrimination in predictions.

As demonstrated by the following Figure on German dataset, the Pearson Correlation of each feature channel to the sensitive feature varies significantly after feature propagation, which further increases the model bias as shown in Figure.
