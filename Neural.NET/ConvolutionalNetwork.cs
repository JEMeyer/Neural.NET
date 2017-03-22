//-----------------------------------------------------------------------
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
    public class ConvolutionalNetwork
    {
        /// <summary>
        /// Constructs a new instance of the <see cref="ConvolutionalNetwork"/> class, only defining
        /// the number of features the network will accept.
        /// </summary>
        public ConvolutionalNetwork()
        {
            // Try and use CUDA. If that fails, try MKL. If that fails, try OpenBLAS. If that fails,
            // they get the slow network.
            bool _temp = Control.TryUseNativeCUDA() || Control.TryUseNativeMKL() || Control.TryUseNativeOpenBLAS();

            this.LayerInformation = new List<ILayerInformation>();
        }

        /// <summary>
        /// Gets or sets the fully connected network
        /// </summary>
        private FullyConnectedNetwork FullyConnectedNetwork { get; set; }

        /// <summary>
        /// Gets or sets the layer information for each layer in the network
        /// </summary>
        private List<ILayerInformation> LayerInformation { get; set; }

        /// <summary>
        /// Adds a convolutional layer to the network.
        /// </summary>
        /// <param name="filterCount">The number of filters this layer should use.</param>
        /// <param name="filterDimension">
        /// A side length of a filter (all filters are square, so if you want a 5x5 filter this
        /// parameter should be 5)
        /// </param>
        /// <param name="stride">The stride used for the filters.</param>
        public void AddConvolutionalLayer(int filterCount, int filterDimension, int stride)
        {
            this.LayerInformation.Add(new ConvolutionalLayerInformation
            {
                FilterCount = filterCount,
                Stride = stride,
                SideLength = filterDimension
            });
        }

        /// <summary>
        /// Adds a fully connected layer to the network. This is the final stage and at least one
        /// layer is REQUIRED. Once you make a fully connected you cannot add other kinds of layers.
        /// </summary>
        /// <param name="nodeCount">The number of nodes in this fully connected layer.</param>
        public void AddFullyConnectedLayer(int nodeCount)
        {
            if (this.FullyConnectedNetwork == null)
            {
                this.FullyConnectedNetwork = new FullyConnectedNetwork(nodeCount);
            }
            else
            {
                this.FullyConnectedNetwork.AddLayer(nodeCount);
            }
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
                SideLength = poolingDimension,
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
                        _currentImages = this.Convolute(_convInfo, _currentImages);
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
        /// Uses the staged layer information to initialize the network.
        /// </summary>
        public void InitializeNetwork()
        {
            foreach (ILayerInformation _layerInformation in this.LayerInformation)
            {
                switch (_layerInformation.LayerType)
                {
                    case (LayerType.Convolutional):
                        ConvolutionalLayerInformation _convInfo = _layerInformation as ConvolutionalLayerInformation;
                        _convInfo.FlattenedFilters = Matrix<double>.Build.Random(_convInfo.FilterCount, (int)Math.Pow(_convInfo.SideLength, 2), new Normal(0.0, 1.0));
                        break;
                }
            }
        }

        /// <summary>
        /// Runs through a single convolution layer.
        /// </summary>
        /// <param name="layerInfo">The layer info used for this layer.</param>
        /// <param name="inputImages">
        /// A matrix of all the images that will be convolved. Each row is an image.
        /// </param>
        /// <returns>A matrix of all the resulting images. Each row is an image.</returns>
        private Matrix<double> Convolute(ConvolutionalLayerInformation layerInfo, Matrix<double> inputImages)
        {
            Matrix<double> _preConvolutionMap = null;
            Matrix<double> _outputImages = null;

            for (int i = 0; i < inputImages.RowCount; i++)
            {
                _preConvolutionMap = this.CreateMaskingMap(layerInfo.SideLength, layerInfo.Stride, inputImages.Row(i));

                _outputImages = _outputImages == null ?
                    _preConvolutionMap :
                    _outputImages.Stack(_preConvolutionMap);
            }

            return _outputImages;
        }

        /// <summary>
        /// Helper function used to create the masking map from a single image represented as a
        /// vector. We do this step so we can do ALL convolutions or pools for an image in one single
        /// computational step.
        /// </summary>
        /// <param name="filterSideLength">
        /// The length of one side of the convolution/pool. All filters are assumed square.
        /// </param>
        /// <param name="strideSize">The stride length that will be used with this map.</param>
        /// <param name="startingImage">
        /// An images represented as a vector that we are creating the map for.
        /// </param>
        /// <returns></returns>
        private Matrix<double> CreateMaskingMap(int filterSideLength, int strideSize, Vector<double> startingImage)
        {
            int _imageSideDimension = (int)Math.Sqrt(startingImage.Count);
            int _endingImageSideDimension = (_imageSideDimension - filterSideLength) / strideSize + 1;
            Matrix<double> _result = CreateMatrix.Dense<double>((int)Math.Pow(filterSideLength, 2), (int)Math.Pow(_endingImageSideDimension, 2));

            for (int i = 0; i < _endingImageSideDimension; i += strideSize)
            {
                for (int j = 0; j < _endingImageSideDimension; j += strideSize)
                {
                    int _arrayIndex = i * _imageSideDimension + j;
                    Vector<double> _dataPatch = Vector<double>.Build.Sparse((int)Math.Pow(filterSideLength, 2));
                    for (int k = 0; k < strideSize; k++)
                    {
                        for (int m = 0; m < filterSideLength; m++)
                        {
                            _dataPatch[k * filterSideLength + m] = startingImage[_arrayIndex + m + filterSideLength * k];
                        }
                    }
                    _result.SetColumn(j + i * filterSideLength, _dataPatch);
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
        private Matrix<double> NonLinear(NonLinearLayerInformation layerInfo, Matrix<double> inputImages)
        {
            switch (layerInfo.NonLinearFunction)
            {
                case NonLinearFunction.Sigmoid:
                    return NonLinearTransformations.Sigmoid(inputImages);

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
        private Matrix<double> Pool(PoolingLayerInformation layerInfo, Matrix<double> inputImages)
        {
            Matrix<double> _preConvolutionMap = null;
            Matrix<double> _outputImages = CreateMatrix.Dense<double>(inputImages.RowCount, (int)Math.Pow((Math.Sqrt(inputImages.ColumnCount) - layerInfo.SideLength) / layerInfo.Stride + 1, 2));

            for (int i = 0; i < inputImages.RowCount; i++)
            {
                _preConvolutionMap = this.CreateMaskingMap(layerInfo.SideLength, layerInfo.Stride, inputImages.Row(i));

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