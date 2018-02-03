## GAN-discovery

A simple example to gain GAN's idea fast.

## Generator Architecture
![](images/gen-architecture.png)

## Discriminator Architecture
![](https://latex.codecogs.com/gif.latex?%5Cmin_G%5Cmax_GV%5Cleft%20%28%20D%2C%20G%20%5Cright%20%29%20%3D%20E_%7Bx%20%5Csim%20p_%7Bdata%7D%28x%29%7D%5BlogD%28x%29%5D%20&plus;%20E_%7Bz%20%5Csim%20p_%7Bz%7D%28z%29%7D%5Blog%281%20-%20D%28G%28z%29%29%29%5D)

![](images/discrim-architecture.png)

## Algorithms

![](images/gan-training.png)

## Reference
1. Dataset
2. [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
3. [Image Completion with Deep Learning in Tensorflow](http://bamos.github.io/2016/08/09/deep-completion)
4. [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/abs/1607.07539)
5. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
6. [Probability Density Function](https://en.wikipedia.org/wiki/Probability_density_function)
7. [DCGANs](https://arxiv.org/abs/1511.06434)
