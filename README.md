# Neural-Networks-from-Scratch
Building neural networks using Python and NumPy based on [Neural Networks from Scratch by Kinsley &amp; Kukiela](https://nnfs.io/).

## Terms

### Data Structures
- Array: a common programming data structure consisting of a homogeneous (same type of data) container of numbers; used in NumPy
- List: Python data structure similar to an array but can contain non-homogenous data.
- Scalar: a single number; rank 0 tensor
- Vector: a 1-dimensional container of numbers; a list in python; an array in NumPy; rank 1 tensor
- Matrix: a 2-dimensional container of numbers; a list of lists in python; a 2D array in NumPy; rank 2 tensor
- Tensor: an N-dimensional; in PyTorch or TensorFlow scalars, vectors, and matrices are all tensors. Tensors can be represented as arrays.
- Dimension: the number of indices needed to locate a specific number in a tensor (for example, a list of lists has 2-dimensions, and 2 numbers are needed to locate a specific number in this tensor); the number of indices can be represented as an array/list/vector/tensor, which means the number of dimensions can be determined by counting the number of items in the array. 

### Operations:
- Dot product: Results in a scalar, known as a "weighted sum." A sum of products of vector elements. Example: vector 1 = [A, B, C]; vector 2 = [X, Y, Z]; dot product of vectors 1 and 2 = A*X + B*Y + C*Z.
- Vector addition: element-wise addition. Example: vector 1 = [A, B, C]; vector 2 = [X, Y, Z]; addition of vectors 1 and 2 = [A+X, B+Y, C+Z].
- Matrix product (aka matrix multiplication): a matrix created from two matrices (A,B) by performing dot products of all combinations of rows from matrix A and columns from matrix B. Note: the second dimension of matrix A must match the first dimension of matrix B; if A has a shape of (5,4), and B has a shape of (4,7), matrix multiplication is possible because the inner numbers match: 4. The shape of the resulting matrix is the first dimension of A and the second dimension of B; if the shape of A is (8,5) and the shape of B is (5,4), then the shape of the product matrix will be (8,4).
- Transposition: modifies a matrix so that its rows become columns and columns become rows. 

### Neural Networks:
- Activation function: a function applied to the output of a neuron that modifies the output, usually to make the output non-linear. 
- Dense layer: a fully-connected layer of neurons. 
- Forward pass: passing data through a model from beginning to end. 

## Neurons and Layers

### Single Neuron

Example inputs:

    inputs = [1.0, 2.0, 3.0, 2.5]

Example neuron weights and bias: 

    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

Note: A dot product operation can be performed on the inputs and weights because they both have the same number of elements (each input is connected to the neuron with a weight).  

*Calculate output using basic Python:*

    output = sum(x * w for x, w in zip(inputs, weights)) + bias
    
Calculate output using Python & NumPy:

    output = np.dot(weights, inputs) + bias

### Neuron Layer

Example inputs:

    inputs = [1.0, 2.0, 3.0, 2.5]

Example weights and biases:

    weights = [
        [0.2,0.8,-0.5,1.0],
        [0.5,-0.91,0.26,-0.5],
        [-0.26,-0.27,0.17,0.87]
        ]
    biases = [2.0,3.0,0.5]

Note: This example consists of 4 input values fully connected to 3 neurons. 

Calculate the layer output using basic Python:

    # Output of current layer
    layer_outputs = []
    # For each neuron
    for neuron_weights, neuron_bias in zip(weights, biases):
        # Initialize neuron output to 0
        neuron_output = 0
        # For each input and weight to the neuron
        for n_input, weight in zip(inputs, neuron_weights):
            # Multiply weight and input and add to neuron's output
            neuron_output += n_input*weight
        # Add bias
        neuron_output += neuron_bias
        # Add neuron output to layer's output list
        layer_outputs.append(neuron_output)

 Calculate the layer output using Python & NumPy:

    layer_outputs = np.dot(weights, inputs) + biases

Note: Using NumPy, we can generate the layer output array using 1 line of code instead of 7 lines. 

### Batches
Training in batches of samples rather than individual samples is faster/more efficient with parallel processing and increases stability/generalizability in the training process. 

To calculate the outputs of a layer of neurons with a batch of samples, we use the **matrix product** operation. The matrix product operations takes two matrices and produces a single matrix: 
- The batch matrix (Inputs) is organized so the first dimension contains the Samples (Batch Size) and the second dimension contains the Features.
- The layer matrix (Weights) needs to be organized (for the calculation) so the first dimension contains the features and the second dimension contains the neurons.
- Once the matrix multiplication is performed, the resulting matrix represents the Samples in the first dimension and the Neurons (outputs) in the second dimension. The features have been summed up through the dot product calculation. The resulting matrix shows the output of each neuron in the layer for each sample. 

### Tranposition
The layer matrix is typically stored as a list of lists with the neurons in the first dimension the weights in the second dimension, as shown in the example layer above. To perform the matrix multiplication operation, these dimensions need to be flipped. To do this, we perform a **transposition** operation. 

Example batch of inputs: 

    inputs = [
    [1.0,2.0,3.0,2.50],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
    ]

Calculate the layer outputs using the batch of inputs with NumPy:

    outputs = np.dot(inputs, np.array(weights).T) + biases

Note: The NumPy transpose operation only works on a NumPy array, which is why the weights are converted into a NumPy array above. 

### Multiple Layers
When there are multiple layers, the output of one layer becomes the input to the next layer. As mentioned above, the output matrix for a layer consists of the samples in the first dimension (rows), and the neurons in the second dimension (columns). Each neuron is fully connected to the neurons in the next layer, with each connection represented by a weight. So, when performing the matrix multiplication between the layers, the second layer's first dimension must contain the weights, which match the number of neurons in the previous layer. For this reason, we again must perform a transposition. 

Example calculating the output of two fully-connected layers:

    layer_1_outputs = np.dot(inputs, np.array(weights).T) + biases
    layer_2_outputs = np.dot(layer_1_outputs, np.array(weights2).T) + biases2
    
### Dense Layer Class

To make it easier to build, we will take an object-oriented approach. First, we will define the Dense Layer Class. 

A **dense layer** is a fully-connected layer, in which each input is connected to each neuron in the layer. 

Let's start by defining an initialization method. 

    # Dense layer
    class Layer_Dense:

        # Layer initialization 
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))

Notes:

- Here, the weights are initialized randomly, which works well when training a model from scratch. If we wanted to train a pre-trained model, we would instead initialize the weights and biases to the pre-trained model values.
- np.random.randn produces a Gaussian distribution with a mean of 0 and a variance of 1, so the values will typically be between -1 and 1.
- The shape of the np.random.randn function is determined by the parameters n_inputs, n_neurons.
- We multiply the weights by 0.01 so that the weights will be easier to adjust during the training process. 
- We initialize the weights to be (inputs, neurons) so that we do not need to transpose as described above.
- For now, the biases are initialized to values of 0, which is a common practice, but sometimes we may want to use different values. 

Next we define the forward method, which contains the logic for the forward pass through the network:

    # forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

## Activation Functions

List of activation functions:

- Step: activates with a value of 1 if the neuron's output is greater than 0.
- Linear: the output value equal the input, often used in last layer for regression.
- Sigmoid: returns a value in the range of 0 for negative infinity, 0.5 for the input 0, and 1 for positive infinity.
- Rectified linear: linear for outputs greater than 0, and 0 if the output is less than or equal to 0. Widely used for speed and efficiency.
- Softmax: Often used for classification, it produces a normalized distribution of probabilities (aka "confidence scores"). 

### Rectified Linear Activation Class

    # ReLU activation
    class Activation_ReLU:

        # Forward pass
        def forward(self,inputs):
            # Calculate output values from input
            self.output = np.maximum(0, inputs)

###  Softmax Activation Class

    # Softmax activation 
    class Activation_Softmax:

        # Forward pass
        def forward(self, inputs):

            # Get unnormalized probabilities
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

            # Normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

            self.output = probabilities

Notes:

- There are two steps to the softmax. First, we exponentiate all values. Second, we transform each value into a probability by dividing it by the sum of all the values in the layer for each sample. 
- For the first step, we use NumPy's exp function.
- Additionally, to prevent exploding values, we can subtract the largest value from each value, which makes all the values less than or equal to 0; when these are exponentiated, they range from 0 to 1.
- Since the "inputs" is a matrix with the samples in the first dimension(rows) and the layer neurons in the second dimension (columns), we need to specify axis=1 to select the maximum value across the columns for each row. We also use keepdims=True so that everything stays lined up the same way. The result is that we are subtracting a vector of the max values from the values in the matrix by "broadcasting" the vector across the column for each row. 
- For the second step, we divide each value by the sum of the values in the layer to get a probability, ranging from 0 to 1 for each value.
- For the sum operation, we again specify axis=1 because we want to get the sum of the values of the neurons (columns/axis=1) for each row. 

