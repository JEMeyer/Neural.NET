//-----------------------------------------------------------------------
// <copyright file="ConvolutionalLayerInformation.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.LayerInformation
{
    using MathNet.Numerics.LinearAlgebra;
    using Neural.NET.Enums;
    using System.Collections.Generic;

    /// <summary>
    /// Defines the information that a convolutional layer needs to operate
    /// </summary>
    [Serializable]
    internal class ConvolutionalLayerInformation :
        ILayerInformation
    {
        /// <summary>
        /// The number of filters in this layer
        /// </summary>
        public int FilterCount { get; set; }

        /// <summary>
        /// A side length of a filter (all filters are square, if you want a 5x5, set this to 5)
        /// </summary>
        public int KernelSize { get; set; }

        /// <summary>
        /// The <see cref="LayerType"/> of this layer.
        /// </summary>
        public LayerType LayerType => LayerType.Convolutional;

        /// <summary>
        /// The stride of the filter as it convolves the image
        /// </summary>
        public int Stride { get; set; }

        /// <summary>
        /// A matrix of all filters used in this layer
        /// </summary>
        internal List<Matrix<double>> FlattenedFilters { get; set; }
    }
}