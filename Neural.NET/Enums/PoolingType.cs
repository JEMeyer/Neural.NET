//-----------------------------------------------------------------------
// <copyright file="PoolingType.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.Enums;

/// <summary>
///     An enum for the type of pooling the user desires
/// </summary>
public enum PoolingType
{
    /// <summary>
    ///     Default value
    /// </summary>
    Undefined = 0,

    /// <summary>
    ///     In each given patch, take the largest value
    /// </summary>
    MaxPooling = 1,
}
