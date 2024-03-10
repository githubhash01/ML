import numpy as np 
import matplotlib.pyplot as plt
import mnist

""" 
Loading Data 
"""
images = mnist.load_images('train-images-idx3-ubyte') # 60,000 images
labels = mnist.load_labels('train-labels-idx1-ubyte')

def linearise(image):
    return image.reshape(784)

# training data is the pair of linearised images and their labels
training_data = [(linearise(image), label) for image, label in zip(images, labels)]

# a global weight matrix W of shape (784, 10)
#W = np.matrix(np.random.rand(784, 10) * 0.01)
# intialsing the matrix to zeros
W = np.zeros((784, 10))

"""
Showing the first image


# first image
image1, label1 = training_data[0]
print(label1) # 5
# reshape the image to 28x28
plt.imshow(image1.reshape(28, 28), cmap='gray')
plt.show()
"""

"""
Feedforward: 
    Image -> Linearise -> Weight Matrix -> Output Layer -> Softmax 
"""

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

# softmax with logsumexp trick to avoid overflow and underflow
def softmax_logsumexp(output_layer):
    return np.exp(output_layer - logsumexp(output_layer))

# feedforward function
def feedForward(image): 
    global W
    # linearise the image
    linearised_image = linearise(image) # 784, 1 vector
    # output layer is w^T * x 
    output_layer = np.array(np.dot(W.T, linearised_image)).flatten()
    # softmax function on the output layer to get the probabilities
    probabilities = softmax_logsumexp(output_layer)
    return probabilities

# given the ground truth, calculate the log loss
def logLoss(probabilities, label):
    return -np.log(probabilities[label] + 1e-10)

def avgLogLoss():
    avg_loss = 0
    for sample in training_data:
        x, y = sample
        avg_loss += logLoss(feedForward(x), y)
    return avg_loss / len(training_data)

def update_weights(image, ground_truth, learning_rate):
    global W
    # create a new (784, 10) matrix to store the updates
    updates = np.zeros((784, 10))
    probabilities = feedForward(image)
    for digit in range(10):
        if digit == ground_truth:
            updates[:, digit] = (probabilities[digit] - 1) * image
        else:
            updates[:, digit] = probabilities[digit] * image
    
    W = W - learning_rate * updates

def epoch(learning_rate):
    for sample in training_data:
        image, ground_truth = sample
        update_weights(image, ground_truth, learning_rate)


for i in range(10):
    print(avgLogLoss())
    epoch(0.001)
