# Neural.NET
.NET libarary used to implement and train neural networks using C#

For a full demo, including how to package MNIST data for the network, this repo shows usage of the library: https://github.com/JEMeyer/Neural.NET-MNIST
* Look at the TestRunner.cs file

# Basic usage
## Creating the network
The actual Network class only has one function you can call, and that is the FeedFoward() function. The only other functionality of the Network is to initialize all the weights and biases associated with the neural network. This is all done through the constructor.

The constructor takes three arguments.

    - Number of input nodes (as an int)
    - Number of nodes in each hidden layer (as an array of ints)
    - Number of output nodes (as an int)
    
The length of the hidden layer list dictates how many layers will exist in the network. The nth int in the array will dictate how many nodes will exist in the nth layer of the network.

Here is an example of a "typical" MNIST network with 784 input nodes, and two hidden layers. The first hidden layer has 100 nodes, the next has 50 nodes, and the output layer has 10 nodes.
 > Network _network = new Network(784, new[] { 100, 50 }, 10);
 
With this your network is all initialized. Each layer has the matrix of weights and a vector of biases created to the proper dimensions defined in teh constructor. These matrices and vectors have been initialized with a value between 0.0 and 1.0 following a normal distrubution.

At this point your Network is fully functional, but untrained. You can feed in the values you plan on feeding in after training, but as you'd imagine the results will be more or less random. When you want to run the network, simply use:
> _network.FeedForward(double[] input)

This will return the index of the array in which the network places it's "guess" as to where to classify the input.
 
 ## Training the network

Training has been majorly simplified, just like the Netowrk initialization. All you'll need to do is construct the NetworkTrainer class with the Network object you wish to train, and call the one exposed function.

Here is a typical TestRunner constructor:
> NetworkTrainer _networkTrainer = new NetworkTrainer(_network);

The training function function is called StochasticGradientDescent. This readme will update with any extra learning algorithms used and how to implemenet them.

StochasticGradientDescent takes in 5 parameters:

    - An array of the training data and labels (as double arrays)
    - The numer of epochs or iterations for which to train (as an int)
    - Training mini batch size (as an int - an even divisor of training data count)
    The learning rate (as a double). I recommend starting this around .02-.05
    - The testing data (optional - same array type as training data)

This function will yield a tuple whenever an epoch finishes. That is, when it runs through every training image once, adjusts for error, and then runs through the testing data to report how well it is doing. The tuple will be a pair of of an int (the epoch number) with a double (the success rate of that epoch against the testing data).

Hopefully this allows you to get up and running. Putting it all together, you would be able to create and train a network with these steps:

``` sh
Network _network = new Network(784, new[] { 100, 50 }, 10);
NetworkTrainer _networkTrainer = new NetworkTrainter(_network);
Tupple<double[], double[]> _testData = GetTestData();
Tupple<double[], double[]> _trainingData = GetTrainingData();
foreach (Tuple<int, double?> _result in _trainer.StochasticGradientDescent(
    _trainingData, 1000, 100, .05, _testingData))
{
    Console.WriteLine($"Epoch {_result.Item1}: {_result.Item2}%");
}
```

I hope this is enough to get started. I'll try to keep this up to date while I push out newer Nuget packages. Even if the repo is slightly different I'll end up updating this for the 'bigger' releases.

Once you get the network trained, you can serialize and store it to keep your training progress. Whenever you deserialize, simply use 
```sh
_network.FeedForward(double[] newInput);
```
in order to have your trained network start making predictions.
