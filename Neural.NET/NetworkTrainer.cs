using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Neural.NET
{
    /// <summary>
    /// The class that is used to train a network.
    /// </summary>
    public class NetworkTrainer
    {
        /// <summary>
        /// Constructs a new instance of the NetworkTrainer class that will be used to train the given network.
        /// </summary>
        /// <param name="network">The network to train.</param>
        public NetworkTrainer(Network network)
        {
            this.Network = network;
        }

        /// <summary>
        /// Gets the private syncing object for parallel critical code.
        /// </summary>
        private object _sync => new object();

        /// <summary>
        /// Gets or sets the private network that this trainer is currently training.
        /// </summary>
        private Network Network { get; set; }

        /// <summary>
        /// Train the neural network using mini-batch stochastic gradient descent.
        /// </summary>
        /// <param name="trainingData">A list of tuples "(x, y)" representing the training inputs and the desired outputs.</param>
        /// <param name="epochs">How many iterations will this net run for.</param>
        /// <param name="miniBatchSize">How many input sets will be fed in for each epoch.</param>
        /// <param name="learningRate">The learning rate of the net.</param>
        /// <param name="testData">If "test_data" is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out.</param>
        public IEnumerable<Tuple<int, double?>> StochasticGradientDescent(Tuple<double[], double[]>[] trainingData, int epochs, int miniBatchSize, double learningRate, Tuple<double[], double[]>[] testData = null)
        {
            // Convert the double array into something a we like more (vectors)
            Tuple<Vector<double>, Vector<double>>[] _trainingData = new Tuple<Vector<double>, Vector<double>>[trainingData.Length];
            Tuple<Vector<double>, Vector<double>>[] _testData = testData == null ? null : new Tuple<Vector<double>, Vector<double>>[testData.Length];

            // For each tuple of double arrays, create a tuple of double vectors
            for (int i = 0; i < _trainingData.Length; i++)
            {
                _trainingData[i] = new Tuple<Vector<double>, Vector<double>>(
                    Vector<double>.Build.DenseOfArray(trainingData[i].Item1),
                    Vector<double>.Build.DenseOfArray(trainingData[i].Item2));
            }

            // Start this at 0, it will get set if there is any test data
            int _sizeOfTest = 0;
            if (_testData != null)
            {
                // Just like training, convert the double arrays into double vectors
                for (int i = 0; i < _testData.Length; i++)
                {
                    _testData[i] = new Tuple<Vector<double>, Vector<double>>(
                    Vector<double>.Build.DenseOfArray(testData[i].Item1),
                    Vector<double>.Build.DenseOfArray(testData[i].Item2));
                }

                // This now has a value, so set it
                _sizeOfTest = _testData.Length;
            }

            int _sizeofTraining = _trainingData.Length;

            // This main loop goes through each image once per epoch
            for (int i = 0; i < epochs; i++)
            {
                // Shuffle training data with helper class. Initiate batches array
                new Random().Shuffle(_trainingData);
                Tuple<Vector<double>, Vector<double>>[][] _batches = new Tuple<Vector<double>, Vector<double>>[_sizeofTraining / miniBatchSize][];

                // We need to initialize the tuples to be actual arrays in order for Array.Copy to not throw an argument null exception
                for (int j = 0; j < _batches.Length; j++)
                {
                    _batches[j] = new Tuple<Vector<double>, Vector<double>>[miniBatchSize];
                }

                // We need to segment out the training data into batches
                int _batchesCounter = 0;
                for (int j = 0; j < _sizeofTraining; j += miniBatchSize)
                {
                    // Copies from j to j + miniBatchSet of the training data and puts this array into the _batchCounter-th array of _batches
                    Array.Copy(_trainingData, j, _batches[_batchesCounter++], 0, miniBatchSize);
                }

                // This is the actual updating code. For each batch it will run through all images in the batch and then take the average error to update weights
                foreach (Tuple<Vector<double>, Vector<double>>[] _batch in _batches)
                {
                    this.UpdateMiniBatch(_batch, learningRate);
                }

                // If we have test data, run through that now and yield the results as well as what epoch we are on. If no test data exists, return null for the double (percentage) of error so they still can get what epoch they are on.
                if (_testData != null)
                {
                    yield return new Tuple<int, double?>(i, this.Evaluate(_testData) / _sizeOfTest * 100);
                }
                else
                {
                    yield return new Tuple<int, double?>(i, null);
                }
            }
        }

        /// <summary>
        /// Pass in the features to the NN. These are considered the 'activations' for the first layer of the net
        /// </summary>
        /// <param name="activation">The values for the input layer.</param>
        /// <returns>A vector to give to the next layer of the net.</returns>
        private Vector<double> FeedForward(Vector<double> activation)
        {
            // Take each activation, multiply weights and add bias, then sigmoid it
            for (int i = 0; i < this.Network.LayerCount; i++)
            {
                activation = this.Sigmoid(this.Network._weights[i].Multiply(activation).Add(this.Network._biases[i]));
            }

            return activation;
        }

        /// <summary>
        /// Return the number of test inputs for which the network outputs the correct result.
        /// </summary>
        /// <returns>The number of test inputs for which the network outputs the correct result.</returns>
        /// <param name="testData">Test data to run through the network.</param>
        private double Evaluate(Tuple<Vector<double>, Vector<double>>[] testData)
        {
            // Building this list will help us later so we can run through all the test data at once, and then see how it compated to actual results
            List<Tuple<int, int>> _testResults = new List<Tuple<int, int>>(testData.Length);

            foreach (Tuple<Vector<double>, Vector<double>> _singleRun in testData)
            {
                // For each run we store off the maximum index of what the network thought was the answer, as well as the actual answer
                _testResults.Add(new Tuple<int, int>(this.FeedForward(_singleRun.Item1).MaximumIndex(), _singleRun.Item2.MaximumIndex()));
            }

            // We return back the count of how many were correct. This is a double so we can make it a percentage easier.
            return _testResults.Where(a => a.Item1 == a.Item2).Count();
        }

        /// <summary>
        /// Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        /// </summary>
        /// <param name="batch">An array of tuples represeting an image and it's classification.</param>
        /// <param name="learningRate">The learning rate to use when training.</param>
        private void UpdateMiniBatch(Tuple<Vector<double>, Vector<double>>[] batch, double learningRate)
        {
            Vector<double>[] _nablaB = new Vector<double>[this.Network._biases.Length];
            Matrix<double>[] _nablaW = new Matrix<double>[this.Network._weights.Length];

            // Build weight nabla (del) with Zeroes
            for (int i = 0; i < this.Network._biases.Length; i++)
            {
                _nablaB[i] = Vector<double>.Build.Sparse(this.Network._biases[i].Count, 0);
            }

            // Do same for weights
            for (int i = 0; i < this.Network._weights.Length; i++)
            {
                _nablaW[i] = Matrix<double>.Build.Sparse(this.Network._weights[i].RowCount, this.Network._weights[i].ColumnCount, 0);
            }

            // We get the delta for each pair in the batch, but don't update the network's weights and biases until after the whole batch is complete.
            // Because batches do a bulk update, we can run the batches in parallel and lock the _nabla arrays.
            Parallel.ForEach(batch, (pair) =>
            {
                // Item 1 is for biases, item2 for weights
                Tuple<Vector<double>[], Matrix<double>[]> _deltas = this.BackPropogation(pair.Item1, pair.Item2);

                // We'll lock this just to be safe during the vector addition
                lock (_sync)
                {
                    // Item 1 is for delta nabla biases, item2 is for delta nabla weights
                    for (int i = 0; i < _nablaB.Length; i++)
                    {
                        _nablaB[i] = _nablaB[i].Add(_deltas.Item1[i]);
                    }

                    //_nablaB = _nablaB.Zip(_deltas.Item1, (nb, dnb) => nb.Add(dnb)).ToArray();
                    for (int i = 0; i < _nablaW.Length; i++)
                    {
                        _nablaW[i] = _nablaW[i].Add(_deltas.Item2[i]);
                    }
                }
            });

            // Update the biases and weights based on the del/nabla values from the back propogation function
            for (int i = 0; i < this.Network.LayerCount; i++)
            {
                this.Network._weights[i] = this.Network._weights[i].Subtract(_nablaW[i].Multiply(learningRate / batch.Length));
            }

            for (int i = 0; i < this.Network.LayerCount; i++)
            {
                this.Network._biases[i] = this.Network._biases[i].Subtract(_nablaB[i].Multiply(learningRate / batch.Length));
            }
        }

        /// <summary>
        /// Backs the propogation.
        /// </summary>
        /// <returns>The propogation.</returns>
        /// <param name="input">Input.</param>
        /// <param name="output">Output.</param>
        private Tuple<Vector<double>[], Matrix<double>[]> BackPropogation(Vector<double> input, Vector<double> expectedOutput)
        {
            Vector<double>[] _nablaB = new Vector<double>[this.Network._biases.Length];
            Matrix<double>[] _nablaW = new Matrix<double>[this.Network._weights.Length];

            // Feed forward
            Vector<double> _activation = input;

            // List to store all activations, layer by layer
            List<Vector<double>> _activations = new List<Vector<double>>(this.Network.LayerCount);
            _activations.Add(_activation);

            // Store all the z vectors (outputs of each layer before running sigmoid)
            List<Vector<double>> _zOutputs = new List<Vector<double>>(this.Network.LayerCount);

            // Loop through all layers, store off values before and after sigmoid function. This is basically a feed forward run of the network with the values getting stored off for correction later
            for (int i = 0; i < this.Network.LayerCount; i++)
            {
                Vector<double> _z = this.Network._weights[i].Multiply(_activation).Add(this.Network._biases[i]);
                _zOutputs.Add(_z);
                _activation = this.Sigmoid(_z);
                _activations.Add(_activation);
            }

            // Now to run through the network backwards instead of forward in order to do the correction
            // Since activations also holds the very first layer, it actually has layerCount + 1 entries, so last is layercount, second to last is layercount - 1
            Vector<double> _delta = this.CostDerivative(_activations[this.Network.LayerCount], expectedOutput).PointwiseMultiply(this.Sigmoid(_zOutputs.Last(), true));
            _nablaB[this.Network.LayerCount - 1] = _delta;
            _nablaW[this.Network.LayerCount - 1] = _delta.ToColumnMatrix().TransposeAndMultiply(_activations[this.Network.LayerCount - 1].ToColumnMatrix());

            // Start the loop counting backwards, doing the exact same algorithm we did above for all middle layers
            for (int i = this.Network.LayerCount - 2; i > 0; i--)
            {
                _delta = this.Network._weights[i + 1].TransposeThisAndMultiply(_delta).PointwiseMultiply(this.Sigmoid(_zOutputs[i], true));
                _nablaB[i] = _delta;
                _nablaW[i] = _delta.ToColumnMatrix().TransposeAndMultiply(_activations[i].ToColumnMatrix());
            }

            // Do last layer
            _delta = this.Network._weights[1].TransposeThisAndMultiply(_delta).PointwiseMultiply(this.Sigmoid(_zOutputs[0], true));
            _nablaB[0] = _delta;
            _nablaW[0] = _delta.ToColumnMatrix().TransposeAndMultiply(_activations[0].ToColumnMatrix());

            return new Tuple<Vector<double>[], Matrix<double>[]>(_nablaB, _nablaW);
        }

        /// <summary>
        /// Simply performs the cost derivative. Is given a vector and what it should have been, and subtracts them.
        /// </summary>
        /// <param name="actual">The actual value out of a layer.</param>
        /// <param name="expected">The expected value of that layer.</param>
        /// <returns></returns>
        private Vector<double> CostDerivative(Vector<double> actual, Vector<double> expected)
        {
            return actual.Subtract(expected);
        }

        /// <summary>
        /// Applies the sigmoid function to all elements of a vector
        /// </summary>
        /// <param name="vector">The vector holding the values to run the sigmoid operation on.</param>
        /// <returns>A vector with the sigmoid operation ran on all values.</returns>
        private Vector<double> Sigmoid(Vector<double> vector, bool derivative = false)
        {
            // Sigmoid(x) is 1 / (1 + e^-x). The derivative is Sigmoid(x) * (1 - Sigmoid(x))
            return derivative ?
                vector.Map(x => ((1.0 / (1.0 + Math.Exp(-x))) * (1 - (1.0 / (1.0 + Math.Exp(-x))))), Zeros.Include) :
                vector.Map(x => (1.0 / (1.0 + Math.Exp(-x))), Zeros.Include);
        }
    }
}