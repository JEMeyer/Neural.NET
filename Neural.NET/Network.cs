//-----------------------------------------------------------------------
// <copyright file="Network.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET
{
    using System;
    using System.Collections.Generic;
    using MathNet.Numerics;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.Providers.LinearAlgebra;

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
            this.Biases = new Vector<double>[this.LayerCount];
            this.Weights = new Matrix<double>[this.LayerCount];

            Control.TryUseNativeMKL();

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
                this.Biases[i] = Vector<double>.Build.Random(this.NodesPerLayer[i + 1], new Normal(0.0, 1.0));
                this.Weights[i] = Matrix<double>.Build.Random(this.NodesPerLayer[i + 1], this.NodesPerLayer[i], new Normal(0.0, 1.0));
            }
        }

        /// <summary>
        /// Gets the current provider so the user can check to see which one the network decided to use.
        /// </summary>
        public ILinearAlgebraProvider Provider => Control.LinearAlgebraProvider;

        /// <summary>
        /// Gets or sets the number of input nodes in our network
        /// </summary>
        internal int InputNodeCount { get; set; }

        /// <summary>
        /// Gets or sets the value of how many layers exist in the network.
        /// </summary>
        internal int LayerCount { get; set; }

        /// <summary>
        /// Gets or sets the matrix array holding all weights in our network
        /// </summary>
        internal Matrix<double>[] Weights { get; set; }

        /// <summary>
        /// Gets or sets an array that holds how many nodes are in each layer
        /// </summary>
        internal int[] NodesPerLayer { get; set; }

        /// <summary>
        /// Gets or sets the number of output nodes in our network
        /// </summary>
        internal int OutputNodeCount { get; set; }

        /// <summary>
        /// Gets or sets the vector array holding all biases in our network
        /// </summary>
        internal Vector<double>[] Biases { get; set; }

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
                _activation = this.Sigmoid(this.Weights[i].Multiply(_activation).Add(this.Biases[i]));
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