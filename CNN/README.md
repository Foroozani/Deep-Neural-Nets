### CNN
Here there are two different way to classify images (CNN) with different APIs. One using `Tensorflow` and the other is `Pytorch`.

# Neural_Nets
> Convolution Neural Networks 


![CNN-image](https://github.com/Foroozani/Neural_Nets/blob/main/images/CNN.png)

**Why are convolutional neural networks better than other neural networks in processing data such as images and video?**

The reason why Convolutional Neural Networks (CNNs) do so much better than classic neural networks on images and videos is that the convolutional layers take advantage of inherent properties of images.

**1. Convolutions**

- Simple feedforward neural networks _don’t see any order in their inputs_. If you shuffled all your images in the same way, the neural network would have the very same performance it has when trained on not shuffled images.
 
- CNN, in opposition, take advantage of *local spatial coherence of images*. This means that they are able to reduce dramatically the number of operation needed to process an image by using convolution on _patches of adjacent pixels_, because adjacent pixels together are meaningful. We also call that local connectivity. Each map is then filled with the result of the convolution of a small patch of pixels, slid with a window over the whole image.

**2. Pooling layers**

 a) There are also the pooling layers, which downscale the image. This is possible because we retain throughout the network, features that are organized spatially like an image, and thus downscaling them makes sense as reducing the size of the image. \textbf{On classic inputs you cannot downscale a vector}, as there is no coherence between an input and the one next to it.
