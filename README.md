# Semantic Segmentation
### Introduction
In this project, we label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Method
Use VGG-16 architecture to create a fully convolution neural network for semantic segmentation. Train with an Adam optimizer using a learning rate of 1e-4 for 45 epochs with a batch size of 2. No augmentation was performed. The total loss decreased from ~40 after the second epoch to ~5 after all 45. Training was performed on CPU.
