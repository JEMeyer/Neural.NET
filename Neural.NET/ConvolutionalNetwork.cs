﻿//-----------------------------------------------------------------------
// <copyright file="ConvolutionalNetwork.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using Neural.NET.Enums;
    using Neural.NET.LayerInformation;

    /// <summary>
    /// Class the defines a convolutional network.
    /// </summary>
    [Serializable]
    public class ConvolutionalNetwork
    {
        /// <summary>
        /// Constructs a new instance of the <see cref="ConvolutionalNetwork"/> class, only defining
        /// the number of features the network will accept.
        /// </summary>
        public ConvolutionalNetwork(int inputDimensions)
        {
            // Try and use CUDA. If that fails, try MKL. If that fails, try OpenBLAS. If that fails,
            // they get the slow network.
            bool _temp = Control.TryUseNativeCUDA() || Control.TryUseNativeMKL() || Control.TryUseNativeOpenBLAS();

            this.LayerInformation = new List<ILayerInformation>();
            this.FullyConnectedNetwork = new FullyConnectedNetwork();
            this.InputDimensions = inputDimensions;
        }

        /// <summary>
        /// Gets or sets the fully connected network
        /// </summary>
        internal FullyConnectedNetwork FullyConnectedNetwork { get; set; }

        /// <summary>
        /// Gets or sets the layer information for each layer in the network
        /// </summary>
        internal List<ILayerInformation> LayerInformation { get; set; }

        /// <summary>
        /// How many dimensions the input image will have. 1 for grayscale, 3 for RGB, etc.
        /// </summary>
        private int InputDimensions { get; set; }

        /// <summary>
        /// Adds a convolutional layer to the network.
        /// </summary>
        /// <param name="filterCount">The number of filters this layer should use.</param>
        /// <param name="kernelSize">
        /// A side length of each kernel (all kernels are square, so if you want a 5x5 kernel this
        /// parameter should be 5)
        /// </param>
        /// <param name="stride">The stride used for the filters.</param>
        public void AddConvolutionalLayer(int filterCount, int kernelSize, int stride)
        {
            ConvolutionalLayerInformation _previousConvolutionLayer = this.LayerInformation.LastOrDefault(p => p.LayerType == LayerType.Convolutional) as ConvolutionalLayerInformation;
            int _previousDimensions = _previousConvolutionLayer == null ? this.InputDimensions : _previousConvolutionLayer.FilterCount;
            List<Matrix<double>> _flattenedFilters = new List<Matrix<double>>(filterCount);
            for (int i = 0; i < filterCount; i++)
            {
                _flattenedFilters.Add(Matrix<double>.Build.Random(_previousDimensions, (int)Math.Pow(kernelSize, 2), new Normal(0.0, 1.0)));
            }

            this.LayerInformation.Add(new ConvolutionalLayerInformation
            {
                FilterCount = filterCount,
                Stride = stride,
                KernelSize = kernelSize,
                FlattenedFilters = _flattenedFilters
            });
        }

        /// <summary>
        /// Adds a fully connected layer to the network. This is the final stage and at least one
        /// layer is REQUIRED. Once you make a fully connected you cannot add other kinds of layers.
        /// </summary>
        /// <param name="nodeCount">The number of nodes in this fully connected layer.</param>
        /// <param name="activationFunction">The activation function to use in this layer.</param>
        public void AddFullyConnectedLayer(int nodeCount, NonLinearFunction activationFunction = NonLinearFunction.Sigmoid)
        {
            this.FullyConnectedNetwork.AddLayer(nodeCount, activationFunction);
        }

        /// <summary>
        /// Add a non-linear layer to the network.
        /// </summary>
        /// <param name="function">The non-linear function to use in this layer.</param>
        public void AddNonLinearLayer(NonLinearFunction function)
        {
            this.LayerInformation.Add(new NonLinearLayerInformation
            {
                NonLinearFunction = function
            });
        }

        /// <summary>
        /// Adds a pooling layer to the network.
        /// </summary>
        /// <param name="poolingDimension">
        /// A side length of the pool (all pools are square, so if you want a 5x5 pool this parameter
        /// should be 5)
        /// </param>
        /// <param name="stride">The stride used for the pool.</param>
        /// <param name="poolingType">The type of pooling to use.</param>
        public void AddPoolingLayer(int poolingDimension, int stride, PoolingType poolingType)
        {
            this.LayerInformation.Add(new PoolingLayerInformation
            {
                KernelSize = poolingDimension,
                PoolingType = poolingType,
                Stride = stride
            });
        }

        /// <summary>
        /// Runs through the network and makes a prediction.
        /// </summary>
        /// <param name="input">The input image to run the prediction on.</param>
        /// <returns>The values from the final layer. The largest value is the networks prediction.</returns>
        public double[] FeedForward(double[] input)
        {
            Matrix<double> _currentImages = CreateMatrix.DenseOfRowVectors(CreateVector.DenseOfArray<double>(input));

            foreach (ILayerInformation _layerInformation in this.LayerInformation)
            {
                switch (_layerInformation.LayerType)
                {
                    case (LayerType.Convolutional):
                        ConvolutionalLayerInformation _convInfo = _layerInformation as ConvolutionalLayerInformation;
                        _currentImages = this.Convolve(_convInfo, _currentImages);
                        break;

                    case (LayerType.Pooling):
                        PoolingLayerInformation _poolInfo = _layerInformation as PoolingLayerInformation;
                        _currentImages = this.Pool(_poolInfo, _currentImages);
                        break;

                    case (LayerType.NonLinear):
                        NonLinearLayerInformation _nonLinearInfo = _layerInformation as NonLinearLayerInformation;
                        _currentImages = this.NonLinear(_nonLinearInfo, _currentImages);
                        break;
                }
            }

            double[] _fullyConnectedInput = CreateVector.DenseOfEnumerable<double>(_currentImages.EnumerateRows().SelectMany(a => a)).ToArray();

            return this.FullyConnectedNetwork.FeedForward(_currentImages.Enumerate().ToArray());
        }

        /// <summary>
        /// Runs through a single convolution layer.
        /// </summary>
        /// <param name="layerInfo">The layer info used for this layer.</param>
        /// <param name="inputImages">
        /// A matrix of all the images that will be convolved. Each row is an image.
        /// </param>
        /// <returns>A matrix of all the resulting images. Each row is an image.</returns>
        internal Matrix<double> Convolve(ConvolutionalLayerInformation layerInfo, Matrix<double> inputImages)
        {
            // Construct out return matrix that will include all layers of our images.
            Matrix<double> _outputImages = null;

            foreach (Tuple<int, Vector<double>> _imageDimensionAndIndex in inputImages.EnumerateRowsIndexed())
            {
                // Create the matrix so we can do all of the convolutions at once.
                Matrix<double> _preConvolutionMap = this.CreateMaskingMap(layerInfo.KernelSize, layerInfo.Stride, _imageDimensionAndIndex.Item2);

                Matrix<double> _filtersForThisDimension = CreateMatrix.Dense<double>(layerInfo.FlattenedFilters.Count, layerInfo.FlattenedFilters[0].ColumnCount);

                foreach (Matrix<double> _filter in layerInfo.FlattenedFilters)
                {
                    _filtersForThisDimension.InsertRow(_imageDimensionAndIndex.Item1, _filter.Row(_imageDimensionAndIndex.Item1));
                }

                // Create the result image matrix if it's not created yet
                if (_outputImages == null)
                {
                    _outputImages = CreateMatrix.Dense<double>(layerInfo.FilterCount, _preConvolutionMap.ColumnCount, 0.0);
                }

                // Store off the result of our filters multiplied by our map. This ends up being every filter passing over
                // the entire image, and returning a dimentions for each kernel in the layer. We sum all the dimensional results in one dimension.
                _outputImages = _outputImages.Add(_filtersForThisDimension.Multiply(_preConvolutionMap));
            }
            
            // Return all the resulting dimensions of the images after convolution
            return _outputImages;
        }

        /// <summary>
        /// Helper function used to create the masking map from a single image represented as a
        /// vector. We do this step so we can do ALL convolutions or pools for an image in one single
        /// computational step.
        /// </summary>
        /// <param name="kernelSideLength">
        /// The length of one side of the convolution/pool. All filters are assumed square.
        /// </param>
        /// <param name="strideSize">The stride length that will be used with this map.</param>
        /// <param name="startingImage">
        /// An images represented as a vector that we are creating the map for.
        /// </param>
        /// <returns></returns>
        internal Matrix<double> CreateMaskingMap(int kernelSideLength, int strideSize, Vector<double> startingImage)
        {
            int _imageSideDimension = (int)Math.Sqrt(startingImage.Count);
            int _endingImageSideDimension = (_imageSideDimension - kernelSideLength) / strideSize + 1;
            Matrix<double> _result = CreateMatrix.Dense<double>((int)Math.Pow(kernelSideLength, 2), (int)Math.Pow(_endingImageSideDimension, 2));

            for (int i = 0; i < _endingImageSideDimension; i += strideSize)
            {
                for (int j = 0; j < _endingImageSideDimension; j += strideSize)
                {
                    int _arrayIndex = i * _imageSideDimension + j;
                    Vector<double> _dataPatch = Vector<double>.Build.Sparse((int)Math.Pow(kernelSideLength, 2));
                    for (int k = 0; k < kernelSideLength; k++)
                    {
                        for (int m = 0; m < kernelSideLength; m++)
                        {
                            _dataPatch[k * kernelSideLength + m] = startingImage[_arrayIndex + m + kernelSideLength * k];
                        }
                    }
                    _result.SetColumn(j + i * _endingImageSideDimension, _dataPatch);
                }
            }

            return _result;
        }

        /// <summary>
        /// Runs through a non-linear layer of the network.
        /// </summary>
        /// <param name="layerInfo">The layer info used for this layer.</param>
        /// <param name="inputImages">
        /// A matrix of all the images that will be ran through the non-linear function. Each row is
        /// an image.
        /// </param>
        /// <returns>A matrix of all the resulting images. Each row is an image.</returns>
        internal Matrix<double> NonLinear(NonLinearLayerInformation layerInfo, Matrix<double> inputImages)
        {
            switch (layerInfo.NonLinearFunction)
            {
                case NonLinearFunction.Sigmoid:
                    return NonLinearTransformations.Sigmoid(inputImages);

                case NonLinearFunction.Tanh:
                    return NonLinearTransformations.Tanh(inputImages);

                case NonLinearFunction.ReLU:
                    return NonLinearTransformations.ReLU(inputImages);

                case NonLinearFunction.LReLU:
                    return NonLinearTransformations.LReLU(inputImages);

                default:
                    return null;
            }
        }

        /// <summary>
        /// Runs through a single pooling layer.
        /// </summary>
        /// <param name="layerInfo">The layer info used for this layer.</param>
        /// <param name="inputImages">
        /// A matrix of all the images that will be pooled. Each row is an image.
        /// </param>
        /// <returns>A matrix of all the resulting images. Each row is an image.</returns>
        internal Matrix<double> Pool(PoolingLayerInformation layerInfo, Matrix<double> inputImages)
        {
            Matrix<double> _preConvolutionMap = null;
            Matrix<double> _outputImages = CreateMatrix.Dense<double>(inputImages.RowCount, (int)Math.Pow((Math.Sqrt(inputImages.ColumnCount) - layerInfo.KernelSize) / layerInfo.Stride + 1, 2));

            for (int i = 0; i < inputImages.RowCount; i++)
            {
                _preConvolutionMap = this.CreateMaskingMap(layerInfo.KernelSize, layerInfo.Stride, inputImages.Row(i));

                switch (layerInfo.PoolingType)
                {
                    case (PoolingType.MaxPooling):
                        _outputImages.SetRow(i, _preConvolutionMap.ColumnNorms(double.PositiveInfinity));
                        break;
                }
            }

            return _outputImages;
        }
    }
}