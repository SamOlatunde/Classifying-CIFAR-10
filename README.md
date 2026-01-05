# CIFAR-10 Image Classification with Neural Networks

This repository explores and compares multiple neural network architectures for image classification on the CIFAR-10 dataset. The project evaluates both Fully Connected Neural Networks (FCNNs) and Convolutional Neural Networks (CNNs) to study how architectural choices affect performance, generalization, and training behavior.

## Dataset
- **CIFAR-10**
- 60,000 RGB images (32×32)
- 10 classes (airplanes, automobiles, birds, cats, ships, etc.)
- Images normalized to the range [0, 1]
- 20% validation split used during training

## Models Implemented

### Fully Connected Neural Networks (FCNN)
- Multiple variants with increasing depth and width
- Experiments with:
  - Activation functions (Sigmoid, ReLU)
  - Optimizers (SGD, SGD with momentum, RMSprop)
  - Batch size changes
  - Regularization (Dropout, L1)
  - Batch normalization
- FCNN Variant 5 selected as the final baseline model

### Convolutional Neural Networks (CNN)
The following CNN architectures were implemented and evaluated:
- **LeNet (modified)**
- **AlexNet (modified)**
- **VGG16 (modified)**
- **VGG19 (modified)**
- **GoogLeNet / Inception-style network with auxiliary classifiers**

All CNNs preserve spatial structure via convolution and pooling layers and significantly outperform FCNN baselines.

## Training Details
- Framework: **Keras (JAX backend)**
- Loss: Sparse Categorical Cross-Entropy
- Metrics: Accuracy, Precision, Recall, F1 Score
- Optimizers: Adam, Nadam, RMSprop, SGD
- Epochs and batch sizes vary by architecture based on convergence behavior

## Results Summary

| Model      | Accuracy | F1 Score |
|-----------|----------|----------|
| FCNN      | 0.516    | 0.516    |
| LeNet     | 0.523    | 0.513    |
| AlexNet   | 0.666    | 0.664    |
| VGG16     | 0.683    | 0.677    |
| VGG19     | 0.786    | 0.785    |
| GoogLeNet | **0.795** | **0.796** |

- CNNs consistently outperform FCNNs
- Deeper architectures (VGG19, GoogLeNet) achieve the best performance
- GoogLeNet attains the highest accuracy but shows weaker generalization during training

## Key Observations
- FCNNs struggle with image data due to loss of spatial information
- Dropout improves generalization but can reduce training accuracy
- Batch normalization requires careful tuning and can introduce validation noise
- Auxiliary classifiers in GoogLeNet improve convergence but not necessarily generalization

## Future Work
- Explore **ResNet architectures** to address overfitting and vanishing gradients
- Further hyperparameter tuning for batch normalization and regularization
- Investigate data augmentation to improve generalization

## References
- Keras Documentation
- GeeksforGeeks: *Understanding GoogLeNet Model CNN Architecture*

