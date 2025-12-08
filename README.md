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
- Vector additon: element-wise additon. Example: vector 1 = [A, B, C]; vector 2 = [X, Y, Z]; addition of vectors 1 and 2 = [A+X, B+Y, C+Z].

## Building Blocks

### Single Neuron

Example inputs:

    inputs = [1.0, 2.0, 3.0, 2.5]

Example neuron weights and bais: 

    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

Note: A dot prodcut operation can be performed on the inputs and weights becuase they both have the same number of elements (each input is connected to the neuron with a weight).  

*Calculate outpout using basic Python:*

    output = sum(x * w for x, w in zip(inputs, weights)) + bias
    
Calculate outpout using Python & NumPy:

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

Note: Using Numpy, we can generate the layer output array using 1 line of code instead of 7 lines. 


