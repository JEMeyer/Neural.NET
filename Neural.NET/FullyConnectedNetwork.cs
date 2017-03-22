//-----------------------------------------------------------------------
// <copyright file="FullyConnectedNetwork.cs" company="Joseph Meyer (Individual)">
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
    public class FullyConnectedNetwork
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FullyConnectedNetwork"/> class.
        /// </summary>
        /// <param name="numFeatures">Number of features (input nodes) for the net.</param>
        /// <param name="numHiddenNodes">
        /// A <see cref="IReadOnlyList{int}"/> of how many hidden nodes should be in each hidden layer.
        /// </param>
        /// <param name="numOutputNodes">How many nodes should be in the output layer.</param>
        public FullyConnectedNetwork(int numFeatures, IReadOnlyList<int> numHiddenNodes, int numOutputNodes)
        {
            // Try and use CUDA. If that fails, try MKL. If that fails, try OpenBLAS. If that fails,
            // they get the slow network.
            bool _temp = Control.TryUseNativeCUDA() || Control.TryUseNativeMKL() || Control.TryUseNativeOpenBLAS();

            int _layerCount = numHiddenNodes.Count + 1;

            // Initialize the size of our arrays
            this.NodesPerLayer = new List<int>(_layerCount + 1);
            this.Biases = new List<Vector<double>>(_layerCount);
            this.Weights = new List<Matrix<double>>(_layerCount);

            // Fill our NumberOfNodes array
            this.NodesPerLayer.Add(numFeatures);
            for (int i = 1; i < _layerCount; i++)
            {
                this.NodesPerLayer.Add(numHiddenNodes[i - 1]);
            }

            this.NodesPerLayer.Add(numOutputNodes);

            // Need to randomly make a list of vectors for biases and weights. One between each layer.
            for (int i = 0; i < _layerCount; i++)
            {
                this.Biases.Add(Vector<double>.Build.Random(this.NodesPerLayer[i + 1], new Normal(0.0, 1.0)));
                this.Weights.Add(Matrix<double>.Build.Random(this.NodesPerLayer[i + 1], this.NodesPerLayer[i], new Normal(0.0, 1.0)));
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FullyConnectedNetwork"/> class, only
        /// defining the number of features the network will accept.
        /// </summary>
        /// <param name="numFeatures">The number of features (input nodes) for the net.</param>
        public FullyConnectedNetwork(int numFeatures)
        {
            // Try and use CUDA. If that fails, try MKL. If that fails, try OpenBLAS. If that fails,
            // they get the slow network.
            bool _temp = Control.TryUseNativeCUDA() || Control.TryUseNativeMKL() || Control.TryUseNativeOpenBLAS();

            this.NodesPerLayer = new List<int>(new[] { numFeatures });
            this.Biases = new List<Vector<double>>();
            this.Weights = new List<Matrix<double>>();
        }

        /// <summary>
        /// Gets the current provider so the user can check to see which one the network decided to use.
        /// </summary>
        public ILinearAlgebraProvider Provider => Control.LinearAlgebraProvider;

        /// <summary>
        /// Gets or sets the vector array holding all biases in our network
        /// </summary>
        internal List<Vector<double>> Biases { get; set; }

        /// <summary>
        /// Gets or sets the value of how many layers exist in the network.
        /// </summary>
        internal int LayerCount => this.Weights.Count;

        /// <summary>
        /// Gets or sets an array that holds how many nodes are in each layer
        /// </summary>
        internal List<int> NodesPerLayer { get; set; }

        /// <summary>
        /// Gets or sets the matrix array holding all weights in our network
        /// </summary>
        internal List<Matrix<double>> Weights { get; set; }

        /// <summary>
        /// Adds a new layer to the network.
        /// </summary>
        /// <param name="nodeCount">How many nodes that should be in this layer.</param>
        public void AddLayer(int nodeCount)
        {
            this.NodesPerLayer.Add(nodeCount);
            this.Biases.Add(Vector<double>.Build.Random(nodeCount, new Normal(0.0, 1.0)));
            this.Weights.Add(Matrix<double>.Build.Random(nodeCount, this.NodesPerLayer[this.LayerCount], new Normal(0.0, 1.0)));
        }

        /// <summary>
        /// This is the public facing feed forward.
        /// </summary>
        /// <param name="activation">The values for the input layer.</param>
        /// <returns>The index of the largest value from the network (it's guess).</returns>
        public double[] FeedForward(double[] activation)
        {
            Vector<double> _activation = Vector<double>.Build.DenseOfArray(activation);
            for (int i = 0; i < this.LayerCount; i++)
            {
                _activation = NonLinearTransformations.Sigmoid(this.Weights[i].Multiply(_activation).Add(this.Biases[i]));
            }

            return _activation.ToArray();
        }
    }
}