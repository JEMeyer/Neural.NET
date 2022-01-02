//-----------------------------------------------------------------------
// <copyright file="NonLinearLayerInformation.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

using Neural.NET.Enums;

namespace Neural.NET.LayerInformation;

/// <summary>
///     Defines all information needed to use a non-linear layer.
/// </summary>
[Serializable]
internal class NonLinearLayerInformation :
    ILayerInformation
{
    /// <summary>
    ///     The non-linear function to use in this layer.
    /// </summary>
    public NonLinearFunction NonLinearFunction { get; set; }

    /// <summary>
    ///     The <see cref="LayerType" /> of this layer.
    /// </summary>
    public LayerType LayerType => LayerType.NonLinear;
}
