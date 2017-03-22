//-----------------------------------------------------------------------
// <copyright file="NonLinearLayerInformation.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.LayerInformation
{
    using Neural.NET.Enums;

    /// <summary>
    /// Defines all information needed to use a non-linear layer.
    /// </summary>
    internal class NonLinearLayerInformation :
        ILayerInformation
    {
        /// <summary>
        /// The <see cref="LayerType"/> of this layer.
        /// </summary>
        public LayerType LayerType => LayerType.NonLinear;

        /// <summary>
        /// The non-linear function to use in this layer.
        /// </summary>
        public NonLinearFunction NonLinearFunction { get; set; }
    }
}