//-----------------------------------------------------------------------
// <copyright file="FullyConnectedLayerInformation.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.LayerInformation
{
    using Neural.NET.Enums;

    /// <summary>
    /// All of the information needed for a fully connected layer to be properly defined
    /// </summary>
    internal class FullyConnectedLayerInformation :
        ILayerInformation
    {
        /// <summary>
        /// Gets or sets the activation function to use on this layer.
        /// </summary>
        public NonLinearFunction ActivationFunction { get; set; }

        /// <summary>
        /// The <see cref="LayerType"/> of this layer
        /// </summary>
        public LayerType LayerType => LayerType.FullyConnected;

        /// <summary>
        /// The number of nodes in this layer
        /// </summary>
        public int NodeCount { get; set; }
    }
}