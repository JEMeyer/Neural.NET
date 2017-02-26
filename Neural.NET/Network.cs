//-----------------------------------------------------------------------
// <copyright file="Network.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET
{
    using System;
    using System.Collections.Generic;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;

    /// <summary>
    /// The class that represents a neural network.
    /// </summary>
    [Serializable]
    public class Network
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Network"/> class.
        /// </summary>
        /// <param name="numFeatures">Number of features (input nodes) for the net.</param>
        /// <param name="numHiddenNodes">A <see cref="IReadOnlyList{int}"/> of how many hidden nodes should be in each hidden layer.</param>
        /// <param name="numOutputNodes">How many nodes should be in the output layer.</param>
        public Network(int numFeatures, IReadOnlyList<int> numHiddenNodes, int numOutputNodes)
        {
            this.LayerCount = numHiddenNodes.Count + 1;
            this.InputNodeCount = numFeatures;
            this.OutputNodeCount = numOutputNodes;

            // Initialize the size of our arrays
            this.NodesPerLayer = new int[this.LayerCount + 1];
            this._biases = new Vector<double>[this.LayerCount];
            this._weights = new Matrix<double>[this.LayerCount];

            // Fill our NumberOfNodes array
            this.NodesPerLayer[0] = this.InputNodeCount;
            this.NodesPerLayer[this.LayerCount] = this.OutputNodeCount;
            for (int i = 1; i < this.LayerCount; i++)
            {
                this.NodesPerLayer[i] = numHiddenNodes[i - 1];
            }

            // Need to randomly make a list of vectors for biases and weights. One between each layer.
            for (int i = 0; i < this.LayerCount; i++)
            {
                this._biases[i] = Vector<double>.Build.Random(this.NodesPerLayer[i + 1], new Normal(0.0, 1.0));
                this._weights[i] = Matrix<double>.Build.Random(this.NodesPerLayer[i + 1], this.NodesPerLayer[i], new Normal(0.0, 1.0));
            }
        }

        /// <summary>
        /// Gets the public facing biases double array of arrays.
        /// </summary>
        public double[][] Biases
        {
            get
            {
                double[][] _tempBiases = new double[this._biases.Length][];
                for (int i = 0; i < _tempBiases.Length; i++)
                {
                    _tempBiases[i] = this._biases[i].ToArray();
                }
                return _tempBiases;
            }
        }

        /// <summary>
        /// Gets or sets the number of input nodes in our network
        /// </summary>
        public int InputNodeCount { get; set; }

        /// <summary>
        /// Gets or sets the value of how many layers exist in the network.
        /// </summary>
        public int LayerCount { get; set; }

        /// <summary>
        /// Gets or sets the matrix array holding all weights in our network
        /// </summary>
        internal Matrix<double>[] _weights { get; set; }

        /// <summary>
        /// Gets or sets an array that holds how many nodes are in each layer
        /// </summary>
        public int[] NodesPerLayer { get; set; }

        /// <summary>
        /// Gets or sets the number of output nodes in our network
        /// </summary>
        public int OutputNodeCount { get; set; }

        /// <summary>
        /// Gets or sets the vector array holding all biases in our network
        /// </summary>
        internal Vector<double>[] _biases { get; set; }

        /// <summary>
        /// Gets the public getter for weights so the consumer does not need to have mathnet numerics as a dependency.
        /// </summary>
        public double[][,] Weights
        {
            get
            {
                double[][,] _tempWeights = new double[this._weights.Length][,];
                for (int i = 0; i < _tempWeights.Length; i++)
                {
                    _tempWeights[i] = this._weights[i].ToArray();
                }
                return _tempWeights;
            }
        }

        /// <summary>
        /// This is the public facing feed forward.
        /// </summary>
        /// <param name="activation">The values for the input layer.</param>
        /// <returns>The index of the largest value from the network (it's guess).</returns>
        public int FeedForward(double[] activation)
        {
            Vector<double> _activation = Vector<double>.Build.DenseOfArray(activation);
            for (int i = 0; i < this.LayerCount; i++)
            {
                _activation = this.Sigmoid(this._weights[i].Multiply(_activation).Add(this._biases[i]));
            }

            return _activation.MaximumIndex();
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