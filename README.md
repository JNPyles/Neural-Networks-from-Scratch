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

## Building Blocks

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
- The layer matrix (Weights) needs to be organized (for the calculation) so the first dimension contains the features and the second dimension contains the Neurons.
- Once the matrix multiplication is performed, the resulting matrix represents the Samples in the first dimension and the Neurons (outputs) in the second dimension. The features have been summed up through the dot product calculation. The resulting matrix shows the output of each neuron in the layer for each sample. 

#### Tranposition
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


