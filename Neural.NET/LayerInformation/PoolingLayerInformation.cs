//-----------------------------------------------------------------------
// <copyright file="PoolingLayerInformation.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET.LayerInformation
{
    using Neural.NET.Enums;

    /// <summary>
    /// All of the information needed to make a pooling layer.
    /// </summary>
    [Serializable]
    internal class PoolingLayerInformation :
        ILayerInformation
    {
        /// <summary>
        /// A side length of a pool (all pools are square, if you want 5x5 pooling this value should
        /// be 5)
        /// </summary>
        public int KernelSize { get; set; }

        /// <summary>
        /// The <see cref="LayerType"/> for this layer.
        /// </summary>
        public LayerType LayerType => LayerType.Pooling;

        /// <summary>
        /// The type of pooling to use for each pool.
        /// </summary>
        public PoolingType PoolingType { get; set; }

        /// <summary>
        /// The stride the pool should move.
        /// </summary>
        public int Stride { get; set; }
    }
}