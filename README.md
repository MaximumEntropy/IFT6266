# IFT-6266 Class Project

## Conditional Image Generation - Inpainting with MSCOCO

### Introduction

I took a pretty standard approach to solving the problem of inpainting an image. Given a 64 x 64 image from MSCOCO, with it's center (32 x 32) masked out, I built a fully convolutional architecture that attempts to predict the center with an L2 reconstruction loss + a Wasserstein GAN objective. Some relevant literature that I used when building this model was

* Convolutional autoencoders with an L2 reconstruction objective - https://pdfs.semanticscholar.org/1c6d/990c80e60aa0b0059415444cdf94b3574f0f.pdf
* Mixing an L2 reconstruction objective with a GAN objective was presented in Context Encoders: Feature Learning by Inpainting - https://arxiv.org/abs/1604.07379. 
* Wasserstein GANs - https://arxiv.org/abs/1701.07875
* Pix2Pix (Image-to-Image Translation with Conditional Adversarial Networks) - Fully convolutional model with an adverserial objective - https://arxiv.org/abs/1611.07004
* DCGAN (Deep Convolutional Generative Adverserial Networks) - https://arxiv.org/abs/1511.06434
* DenseNet (Denseley Connected Convolutional Networks) - https://arxiv.org/abs/1608.06993
* UNet - https://arxiv.org/abs/1505.04597

An example of masked image  and it's corresponding ground-truth is shown in the image below.

![Task](/images/lamb.png)

### Architecture

I used a fully convolutional architecture for the generator, that is very similar to the DCGAN generator with strided convolution and transpose convolution layers, batch-normalization and ReLU activations. The difference between my generator and the DCGAN generator is that I don't have the initial linear transformation that they use to reshape the noise prior. The discriminator is also similar to the DCGAN discriminator with strided convolutions, batch normalization and ReLUs.

As in the pix2pix paper, I added skip connections between the feature maps produced by the regular convolutions and the transpose convoltion feature maps. The feature maps are concatenated along the channel axis.

#### Generator

| Layer | Filters | Input Shape | Output Shape | Kernel Size |
| ------------- | ------------- | ------------- | ------------ | ------------ |
| Conv1 | 32 | 3 x 64 x 64 | 32 x 32 x 32 | 4 x 4 |
| Conv2 | 64 | 32 x 32 x 32 | 64 x 16 x 16 | 4 x 4 |
| Conv3 | 96 | 64 x 16 x 16 | 96 x 8 x 8 | 4 x 4 |
| Conv4 | 128 | 96 x 8 x 8 | 128 x 4 x 4 | 4 x 4 |
| TConv1 | 96 | 128 x 4 x 4 | 96 x 8 x 8 | 4 x 4 |
| TConv2 | 64 | (96 + 96) x 8 x 8 | 64 x 16 x 16 | 4 x 4 |
| TConv3 | 1 | (64 + 64) x 16 x 16 | 1 x 32 x 32 | 4 x 4 |

The table above presents the generator's architecture where Conv* refers to a regular convolution with a stride of (2, 2) and TConv* refers to a transpose convolution, also with a stride of (2, 2).

#### Discriminator

| Layer | Filters | Input Shape | Output Shape | Kernel Size |
| ------------- | ------------- | ------------- | ------------ | ------------ |
| Conv1 | 32 | 3 x 32 x 32 | 32 x 16 x 16 | 4 x 4 |
| Conv2 | 64 | 32 x 16 x 16 | 64 x 8 x 8 | 4 x 4 |
| Conv3 | 96 | 64 x 8 x 8 | 96 x 4 x 4 | 4 x 4 |
| Conv4 | 128 | 96 x 4 x 4 | 128 x 2 x 2 | 4 x 4 |
| Conv5 | 256 | 128 x 2 x 2 | 256 x 1 x 1 | 2 x 2 |


### Training

As evident from the Generator architecture, the model takes a 3 channel 64 x 64 image with the center masked out and tries to predict the center 32 x 32 square. The discriminator (when using the adverserial objective) takes "fake" 32 x 32 samples from our Generator and "real" 32 x 32 samples from our training data and is trained to distinguish between them. The generator is trained to fool the discriminator. This is formulated as a minimax game as described in Ian Goodfellow's original GAN paper. The L2 reconstruction objective tries to minimize the squared error between the predicted image center and the ground truth. The L2 + GAN objective simply adds these two losses together and backpropagates it through the network. In this work, I used a Wasserstein GAN instead of a regular GAN which from an implementation perspective, simply removes `log` from the GAN objective and clamps the weights of the discriminator.

### Hyperparameters 

- Optimizer - ADAM for both the generator and discriminator with a learning rate of 2e-3
- Discriminator weight clamping -  (-0.03, 0.03) if discriminator is trained 5 times for every generator update else (-0.05, 0.05). 
- Batch size - 32

### Results

In this section, I will present some of my cherry-picked inpainted images.

#### L2

![sample_1](l2_epoch_10_samples.png)
![sample_2](l2_epoch_13_samples.png)
![sample_3](l2_epoch_27_samples.png)
![sample_4](l2_epoch_30_samples.png)

#### L2 + GAN

![sample_1](gan_epoch_22_samples.png)
![sample_2](gan_epoch_27_samples.png)
![sample_3](gan_epoch_50_samples.png)
![sample_4](gan_epoch_54_samples.png)

#### L2 + GAN + Caption
