//-----------------------------------------------------------------------
// <copyright file="RandomExtensions.cs" company="Joseph Meyer (Individual)">
//     Copyright (c) Joseph Meyer. All rights reserved.
// </copyright>
//-----------------------------------------------------------------------

namespace Neural.NET
{
    using System;

    /// <summary>
    /// Class used to hold extension methods of the 'Random' class used by the network.
    /// </summary>
    internal static class RandomExtensions
    {
        /// <summary>
        /// Shuffles an array of values in O(n) time.
        /// </summary>
        /// <typeparam name="T">The type of value the array holds. For our net, will likely be an array of tuples, holding two vectors of doubles</typeparam>
        /// <param name="rng">The internal Random class.</param>
        /// <param name="array">The array to shuffle.</param>
        public static void Shuffle<T>(this Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}