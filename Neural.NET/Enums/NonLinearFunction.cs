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
        Sigmoid = 1
    }
}