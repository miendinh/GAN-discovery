## GAN-discovery

A simple example to gain GAN's idea fast.

## Generator Architecture
![](images/gen-architecture.png)

## Discriminator Architecture
![](images/discrim-architecture.png)

$\min_G\max_GV\left ( D, G \right ) = E_{x \sim p_{data}(x)}[logD(x)] + E_{z \sim p_{z}(z)}[log(1 - D(G(z)))]$

## Algorithms

![](images/gan-training.png)

## Reference
1. Dataset
* [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
* [Image Completion with Deep Learning in Tensorflow](http://bamos.github.io/2016/08/09/deep-completion)
* [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/abs/1607.07539)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
* [Probability Density Function](https://en.wikipedia.org/wiki/Probability_density_function)
* [DCGANs](https://arxiv.org/abs/1511.06434)
