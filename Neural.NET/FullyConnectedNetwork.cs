//-----------------------------------------------------------------------
// <copyright file="FullyConnectedNetwork.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Neural.NET.Enums;
using Neural.NET.LayerInformation;

namespace Neural.NET;

/// <summary>
///     The class that represents a fully connected neural network.
/// </summary>
[Serializable]
public class FullyConnectedNetwork
{
    /// <summary>
    ///     Initializes a new instance of the <see cref="FullyConnectedNetwork" /> class.
    /// </summary>
    public FullyConnectedNetwork()
    {
        // Try and use CUDA. If that fails, try MKL. If that fails, try OpenBLAS. If that fails,
        // they get the slow network.
        var _temp = Control.TryUseNativeCUDA() || Control.TryUseNativeMKL() || Control.TryUseNativeOpenBLAS();

        this.LayerInformation = new List<FullyConnectedLayerInformation>();
        this.Biases = new List<Vector<double>>();
        this.Weights = new List<Matrix<double>>();
    }

    /// <summary>
    ///     Gets the current provider so the user can check to see which one the network decided to use.
    /// </summary>
    public ILinearAlgebraProvider Provider => LinearAlgebraControl.Provider;

    /// <summary>
    ///     Gets or sets the vector array holding all biases in our network
    /// </summary>
    internal List<Vector<double>> Biases { get; set; }

    /// <summary>
    ///     Gets or sets the value of how many layers exist in the network.
    /// </summary>
    internal int LayerCount => this.Weights.Count;

    /// <summary>
    ///     Gets or sets an array that holds information defining each layer.
    /// </summary>
    internal List<FullyConnectedLayerInformation> LayerInformation { get; set; }

    /// <summary>
    ///     Gets or sets the matrix array holding all weights in our network
    /// </summary>
    internal List<Matrix<double>> Weights { get; set; }

    /// <summary>
    ///     Adds a new layer to the network.
    /// </summary>
    /// <param name="nodeCount">How many nodes that should be in this layer.</param>
    /// <param name="activationFunction">The activation function to use for this layer.</param>
    public void AddLayer(int nodeCount, NonLinearFunction activationFunction)
    {
        this.LayerInformation.Add(
            item: new FullyConnectedLayerInformation {NodeCount = nodeCount, ActivationFunction = activationFunction});
        if (this.LayerInformation.Count > 1)
        {
            this.Biases.Add(item: Vector<double>.Build.Random(length: nodeCount, distribution: new Normal(mean: 0.0, stddev: 1.0)));
            this.Weights.Add(item: Matrix<double>.Build.Random(rows: nodeCount,
                columns: this.LayerInformation[index: this.LayerCount].NodeCount,
                distribution: new Normal(mean: 0.0, stddev: 1.0)));
        }
    }

    /// <summary>
    ///     This is the public facing feed forward.
    /// </summary>
    /// <param name="activation">The values for the input layer.</param>
    /// <returns>The index of the largest value from the network (it's guess).</returns>
    public double[] FeedForward(double[] activation)
    {
        var _activation = Vector<double>.Build.DenseOfArray(array: activation);
        for (var i = 0; i < this.LayerCount; i++)
        {
            _activation = this.RunActivation(
                activation: this.Weights[index: i].Multiply(rightSide: _activation).Add(other: this.Biases[index: i]),
                activationFunction: this.LayerInformation[index: i].ActivationFunction);
        }

        return _activation.ToArray();
    }

    /// <summary>
    ///     Runs the defines activation function over the given vector.
    /// </summary>
    /// <param name="activation">The values for the input layer.</param>
    /// <param name="activationFunction">The activation function to use.</param>
    /// <param name="derivative">Whether we want the derivative of the values.</param>
    /// <returns>A vector to give to the next layer of the net.</returns>
    internal Vector<double> RunActivation(Vector<double> activation, NonLinearFunction activationFunction, bool derivative = false)
    {
        switch (activationFunction)
        {
            case NonLinearFunction.Sigmoid:
                return NonLinearTransformations.Sigmoid(vector: activation, derivative: derivative);

            case NonLinearFunction.Tanh:
                return NonLinearTransformations.Tanh(vector: activation, derivative: derivative);

            case NonLinearFunction.ReLU:
                return NonLinearTransformations.ReLU(vector: activation, derivative: derivative);

            case NonLinearFunction.LReLU:
                return NonLinearTransformations.LReLU(vector: activation, derivative: derivative);

            default:
                return null;
        }
    }
}
