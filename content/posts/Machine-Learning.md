---
title: "Machine Learning"
date: 2023-05-18T20:24:41+05:30
---

# Unit -2 Artificial Neural Networks

## Introduction:

<!-- ### Understanding Neural Networks:

Neural networks are a powerful subset of machine learning algorithms inspired by the human brain's neural structure. They are designed to mimic the way our brains process information. Just as the brain consists of interconnected neurons, neural networks are composed of artificial neurons, also known as nodes or units. These nodes work collectively to solve complex problems by learning from data. -->

A neural network is a type of machine-learning algorithm inspired by the structure of the human brain. It consists of interconnected processing nodes, called neurons, that work together to process and analyze complex data. Neural networks can learn from examples and can be trained to recognize patterns in data, making them useful for a variety of tasks such as image and speech recognition, natural language processing, and predictive modeling. They are widely used in artificial intelligence applications and are a fundamental building block for many modern technologies.

We’ll sound pretty complicated, right? Let’s break it down.

A neural network is nothing but a program that is designed to work in a way just like a human brain. Just like our brain has tiny cells called neurons that help us to think and learn, a neural network is made up of tiny parts that work together to solve a particular problem.

Still didn’t understand?

Okay, let me give you a real example. We use Facebook right, when we upload a picture onto Facebook, it automatically suggests the people in the image or your friends to tag them. The model recognizes the images and produces an output of who the person in the picture is.

The neural network analyzes the photo and recognizes specific patterns, such as the shape of a person’s face or the color of their hair. It then matches those patterns to a database of previously tagged images to identify who is in the photo.

You ask why a neural network when we could manually tag ourselves, but what if there are thousands of faces in a single image? That’s where we need it for large datasets.

**Why do we use neural networks:** The simple answer is: to understand the relationship between the input X and the output Y to model the unknown function f.

** Well this is an academic blog which means all the content would mostly be in theoretical form, as our college values the quantity of number of pages written rather than the quality of the content.
To work on a hands on problems refer this [Neural Nets](https://github.com/AayushSameerShah/Neural-Net-Zero-to-Hero-with-Andrej/blob/main/01%20-%20Micrograd/Micrograd%20Foundations.ipynb)

### Representation: 

{{< figure src="https://media5.datahacker.rs/2018/08/Features-image-007-Neural-Network-representation.png" >}}

### Cost Function: 
 A cost function is a measure of how well a machine learning model fits the training data. It is used to train the model by finding the model parameters that minimize the cost function.

The cost function is typically a mathematical function that takes the model parameters and the training data as input and outputs a number. The lower the number, the better the model fits the training data.

There are many different types of cost functions, but some of the most common ones include:

**Mean squared error (MSE):** This is the sum of the squared differences between the predicted values and the actual values.

**Cross-entropy:** This is a measure of the difference between the probability distribution of the predicted values and the probability distribution of the actual values.

**Hinge loss:** This is a measure of the margin between the predicted values and the actual values.

### Gradient Descent:

Gradient descent is an optimization algorithm used to minimize a cost function. The cost function measures how well a model fits the training data. Gradient descent works by iteratively moving the model parameters in the direction of the negative gradient of the cost function. This means that gradient descent is always moving in the direction of the steepest descent, which is the direction that will lead to the lowest cost.

Here is an example of how gradient descent can be used to train a linear regression model to predict the price of a house. The training data consists of the following information:

The size of the house in square feet
The number of bedrooms
The number of bathrooms
The location of the house
The goal is to train a model that can predict the price of a house given its size, number of bedrooms, number of bathrooms, and location.

The first step is to choose an initial set of model parameters. In this case, we will choose the model parameters to be 0.

The next step is to calculate the gradient of the cost function with respect to the model parameters. The gradient of the cost function is the derivative of the cost function with respect to the model parameters. The derivative of the cost function is a vector that tells us how much the cost function will change if we change the model parameters by a small amount.

The final step is to update the model parameters by moving in the direction of the negative gradient. In this case, we will move the model parameters in the direction of the negative gradient by a small amount.

We will repeat the steps above until the cost function converges to a minimum. This means that the model parameters will stop changing and the model will be able to predict the price of a house with a high degree of accuracy.

***This course is ongoing...***



<!-- 
### Layers and Architecture:

Neural networks are organized into layers, with each layer consisting of multiple nodes. The three main types of layers in a neural network are:

Input Layer: This layer receives the initial data or input features. It acts as the entry point of information into the network.

Hidden Layers: These layers, as the name suggests, are not directly observable and lie between the input and output layers. Hidden layers are responsible for extracting and learning complex patterns and representations from the input data.

Output Layer: The final layer of a neural network produces the desired output. The number of nodes in this layer depends on the nature of the problem. For example, a neural network used for image classification may have nodes representing different classes.

### Connections and Weights:

Neurons within a neural network are interconnected through connections, often represented as weighted edges. Each connection has an associated weight that determines its strength or importance. During the learning process, these weights are adjusted to optimize the network's performance.

### Activation Function:

An activation function introduces non-linearity to the neural network, allowing it to model complex relationships between inputs and outputs. It determines the output of a node based on its weighted sum of inputs. Popular activation functions include the sigmoid function, ReLU (Rectified Linear Unit), and tanh (hyperbolic tangent). 

### Example: Image Classification

To illustrate the power of neural networks, let's consider an example of image classification. Suppose we have a dataset of handwritten digits (0-9), and our goal is to build a model that can correctly identify the digits from new, unseen images.

We can train a neural network with an input layer receiving pixel values of the images, multiple hidden layers responsible for learning intricate patterns in the images, and an output layer representing the digit classes. The network learns from thousands of labeled images, adjusting the weights of connections to minimize the prediction errors.

Once trained, our neural network can make accurate predictions on unseen images, correctly identifying the handwritten digits. This demonstrates how neural networks excel at complex pattern recognition tasks. -->





