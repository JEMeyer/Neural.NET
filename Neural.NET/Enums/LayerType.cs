//-----------------------------------------------------------------------
// <copyright file="LayerType.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.Enums;

/// <summary>
///     Enum to define the different types of layers a user can add to a network
/// </summary>
internal enum LayerType
{
    /// <summary>
    ///     Default value
    /// </summary>
    Undefined = 0,

    /// <summary>
    ///     A layer that will convolve every input images with various filters
    /// </summary>
    Convolutional = 1,

    /// <summary>
    ///     A layer that will pool values in patches
    /// </summary>
    Pooling = 2,

    /// <summary>
    ///     A layer that computes a non linear function on every data point
    /// </summary>
    NonLinear = 3,

    /// <summary>
    ///     A layer of fully connected neurons
    /// </summary>
    FullyConnected = 4,
}
