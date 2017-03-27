//-----------------------------------------------------------------------
// <copyright file="NonLinearFunction.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.Enums
{
    /// <summary>
    /// An enum that defines what non-linear function is desired
    /// </summary>
    public enum NonLinearFunction
    {
        /// <summary>
        /// Default value
        /// </summary>
        Undefined = 0,

        /// <summary>
        /// The sigmoid function is used as the non-linear function
        /// </summary>
        Sigmoid = 1,

        /// <summary>
        /// The tanh function is used as the non-linear function
        /// </summary>
        Tanh = 2,

        /// <summary>
        /// The rectified linear unit is used as the non-linear function
        /// </summary>
        ReLU = 3,

        /// <summary>
        /// The leaky rectified linear unit is used as the non-linear function
        /// </summary>
        LReLU = 4
    }
}