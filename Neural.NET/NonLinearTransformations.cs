//-----------------------------------------------------------------------
// <copyright file="NonLinearTransformations.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET
{
    using System;
    using MathNet.Numerics.LinearAlgebra;

    internal static class NonLinearTransformations
    {
        /// <summary>
        /// Applies the sigmoid function to all elements of a vector
        /// </summary>
        /// <param name="vector">The vector holding the values to run the sigmoid operation on.</param>
        /// <param name="derivative">Whether we want the derivative</param>
        /// <returns>A vector with the sigmoid operation ran on all values.</returns>
        internal static Vector<double> Sigmoid(Vector<double> vector, bool derivative = false)
        {
            // Sigmoid(x) is 1 / (1 + e^-x). The derivative is Sigmoid(x) * (1 - Sigmoid(x))
            if (derivative)
            {
                Vector<double> _sigmoid = Sigmoid(vector);
                return _sigmoid.PointwiseMultiply(_sigmoid.Subtract(1));
            }
            else
            {
                //vector.Map(x => ((1.0 / (1.0 + Math.Exp(-x))) * (1 - (1.0 / (1.0 + Math.Exp(-x))))), Zeros.Include) :
                return vector.Map(x => (1.0 / (1.0 + Math.Exp(-x))), Zeros.Include);
            }
        }

        /// <summary>
        /// Applies the sigmoid function to all elements of a matrix
        /// </summary>
        /// <param name="vector">The vector holding the values to run the sigmoid operation on.</param>
        /// <param name="derivative">Whether we want the derivative</param>
        /// <returns>A vector with the sigmoid operation ran on all values.</returns>
        internal static Matrix<double> Sigmoid(Matrix<double> matrix, bool derivative = false)
        {
            // Sigmoid(x) is 1 / (1 + e^-x). The derivative is Sigmoid(x) * (1 - Sigmoid(x))
            if (derivative)
            {
                Matrix<double> _sigmoid = Sigmoid(matrix);
                return _sigmoid.PointwiseMultiply(_sigmoid.SubtractFrom(1));
            }
            else
            {
                return matrix.Map(x => (1.0 / (1.0 + Math.Exp(-x))), Zeros.Include);
            }
        }

        /// <summary>
        /// Applies the tanh function to all elements of a vactor
        /// </summary>
        /// <param name="vector">The vector holding the values to run the tanh operation on.</param>
        /// <param name="derivative">Whether we want the derivative</param>
        /// <returns>A vector with the tanh operation ran on all values.</returns>
        internal static Vector<double> Tanh(Vector<double> vector, bool derivative = false)
        {
            // Tanh is provided in the libary. The derivative is 1 - tanh(x)^2
            if (derivative)
            {
                Vector<double> _tanh = vector.PointwiseTanh();
                return _tanh.PointwisePower(2).SubtractFrom(1);
            }
            else
            {
                return vector.PointwiseTanh();
            }
        }

        /// <summary>
        /// Applies the tanh function to all elements of a matrix
        /// </summary>
        /// <param name="vector">The vector holding the values to run the tanh operation on.</param>
        /// <param name="derivative">Whether we want the derivative</param>
        /// <returns>A vector with the tanh operation ran on all values.</returns>
        internal static Matrix<double> Tanh(Matrix<double> matrix, bool derivative = false)
        {
            // Tanh is provided in the libary. The derivative is 1 - tanh(x)^2
            if (derivative)
            {
                Matrix<double> _tanh = matrix.PointwiseTanh();
                return _tanh.PointwisePower(2).SubtractFrom(1);
            }
            else
            {
                return matrix.PointwiseTanh();
            }
        }
    }
}