//-----------------------------------------------------------------------
// <copyright file="NonLinearTransformations.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra;

namespace Neural.NET;

/// <summary>
///     Static class to compute any of the nonlinear functions used in this libaray.
/// </summary>
internal static class NonLinearTransformations
{
    /// <summary>
    ///     Applies the Leaky ReLU (LReLU) function on all elements. Values greater than 0 stay the
    ///     same. Values smaller become 0.
    /// </summary>
    /// <param name="vector">The vector holding the values to run the ReLU operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A vector with the ReLU operation ran on all values.</returns>
    internal static Vector<double> LReLU(Vector<double> vector, bool derivative = false)
    {
        // LReLU is x => x > 0 ? x : .01
        if (derivative)
        {
            return vector.Map(f: x => x > 0 ? 1.0 : 0.0, zeros: Zeros.Include);
        }

        return vector.Map(f: x => x > 0 ? x : 0.01, zeros: Zeros.Include);
    }

    /// <summary>
    ///     Applies the Leaky ReLU (LReLU) function on all elements. Values greater than 0 stay the
    ///     same. Values smaller become 0.
    /// </summary>
    /// <param name="matrix">The matrix holding the values to run the ReLU operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A matrix with the ReLU operation ran on all values.</returns>
    internal static Matrix<double> LReLU(Matrix<double> matrix, bool derivative = false)
    {
        // LReLU is x => x > 0 ? x : 0.01
        if (derivative)
        {
            return matrix.Map(f: x => x > 0 ? 1.0 : 0.0, zeros: Zeros.Include);
        }

        return matrix.Map(f: x => x > 0 ? x : 0.01, zeros: Zeros.Include);
    }

    /// <summary>
    ///     Applies the ReLU function on all elements. Values greater than 0 stay the same. Values
    ///     smaller become 0.
    /// </summary>
    /// <param name="vector">The vector holding the values to run the ReLU operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A vector with the ReLU operation ran on all values.</returns>
    internal static Vector<double> ReLU(Vector<double> vector, bool derivative = false)
    {
        // ReLU is x => x > 0 ? x : 0
        if (derivative)
        {
            return vector.Map(f: x => x > 0 ? 1.0 : 0.0, zeros: Zeros.Include);
        }

        return vector.Map(f: x => x > 0 ? x : 0, zeros: Zeros.Include);
    }

    /// <summary>
    ///     Applies the ReLU function on all elements. Values greater than 0 stay the same. Values
    ///     smaller become 0.
    /// </summary>
    /// <param name="matrix">The matrix holding the values to run the ReLU operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A matrix with the ReLU operation ran on all values.</returns>
    internal static Matrix<double> ReLU(Matrix<double> matrix, bool derivative = false)
    {
        // ReLU is x => x > 0 ? x : 0
        if (derivative)
        {
            return matrix.Map(f: x => x > 0 ? 1.0 : 0.0, zeros: Zeros.Include);
        }

        return matrix.Map(f: x => x > 0 ? x : 0.0, zeros: Zeros.Include);
    }

    /// <summary>
    ///     Applies the sigmoid function to all elements of a vector
    /// </summary>
    /// <param name="vector">The vector holding the values to run the sigmoid operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A vector with the sigmoid operation ran on all values.</returns>
    internal static Vector<double> Sigmoid(Vector<double> vector, bool derivative = false)
    {
        // Sigmoid(x) is 1 / (1 + e^-x). The derivative is Sigmoid(x) * (1 - Sigmoid(x))
        if (derivative)
        {
            var _sigmoid = Sigmoid(vector: vector);
            return _sigmoid.PointwiseMultiply(other: _sigmoid.Subtract(scalar: 1));
        }

        //vector.Map(x => ((1.0 / (1.0 + Math.Exp(-x))) * (1 - (1.0 / (1.0 + Math.Exp(-x))))), Zeros.Include) :
        return vector.Map(f: x => 1.0 / (1.0 + Math.Exp(d: -x)), zeros: Zeros.Include);
    }

    /// <summary>
    ///     Applies the sigmoid function to all elements of a matrix
    /// </summary>
    /// <param name="matrix">The matrix holding the values to run the sigmoid operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A vector with the sigmoid operation ran on all values.</returns>
    internal static Matrix<double> Sigmoid(Matrix<double> matrix, bool derivative = false)
    {
        // Sigmoid(x) is 1 / (1 + e^-x). The derivative is Sigmoid(x) * (1 - Sigmoid(x))
        if (derivative)
        {
            var _sigmoid = Sigmoid(matrix: matrix);
            return _sigmoid.PointwiseMultiply(other: _sigmoid.SubtractFrom(scalar: 1));
        }

        return matrix.Map(f: x => 1.0 / (1.0 + Math.Exp(d: -x)), zeros: Zeros.Include);
    }

    /// <summary>
    ///     Applies the tanh function to all elements of a vactor
    /// </summary>
    /// <param name="vector">The vector holding the values to run the tanh operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A vector with the tanh operation ran on all values.</returns>
    internal static Vector<double> Tanh(Vector<double> vector, bool derivative = false)
    {
        // Tanh is provided in the libary. The derivative is 1 - tanh(x)^2
        if (derivative)
        {
            var _tanh = vector.PointwiseTanh();
            return _tanh.PointwisePower(exponent: 2).SubtractFrom(scalar: 1);
        }

        return vector.PointwiseTanh();
    }

    /// <summary>
    ///     Applies the tanh function to all elements of a matrix
    /// </summary>
    /// <param name="matrix">The matrix holding the values to run the tanh operation on.</param>
    /// <param name="derivative">Whether we want the derivative</param>
    /// <returns>A matrix with the tanh operation ran on all values.</returns>
    internal static Matrix<double> Tanh(Matrix<double> matrix, bool derivative = false)
    {
        // Tanh is provided in the libary. The derivative is 1 - tanh(x)^2
        if (derivative)
        {
            var _tanh = matrix.PointwiseTanh();
            return _tanh.PointwisePower(exponent: 2).SubtractFrom(scalar: 1);
        }

        return matrix.PointwiseTanh();
    }
}
