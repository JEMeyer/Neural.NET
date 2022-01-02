//-----------------------------------------------------------------------
// <copyright file="ILayerInformation.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

using Neural.NET.Enums;

namespace Neural.NET.LayerInformation;

/// <summary>
///     Defines a contract for anything claiming to be layer information
/// </summary>
internal interface ILayerInformation
{
    /// <summary>
    ///     The type of layer the information is for
    /// </summary>
    LayerType LayerType { get; }
}
