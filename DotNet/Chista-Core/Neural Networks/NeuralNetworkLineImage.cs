using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class NeuralNetworkLineImage : INeuralNetworkImage, INeuralNetworkInformation
    {
        public readonly int index;
        public readonly NeuralNetworkImage[] images;
        public readonly IDataCombiner[] combiners;

        public NeuralNetworkLineImage(
            NeuralNetworkImage[] images, IDataCombiner[] combiners, int train_index)
        {
            this.images = images ?? throw new ArgumentNullException(nameof(images));
            this.combiners = combiners ?? throw new ArgumentNullException(nameof(combiners));

            if (images.Length < 2)
                throw new ArgumentOutOfRangeException(nameof(images),
                    "The count of brains must be greater than two");

            if (combiners.Length != images.Length - 1)
                throw new ArgumentOutOfRangeException(nameof(combiners),
                    "The count of brains and combiners are not matched");

            if (train_index >= images.Length)
                throw new ArgumentOutOfRangeException(nameof(train_index));

            index = train_index < 0 ? combiners.Length : train_index;
        }

        public override string ToString()
        {
            var buffer = new StringBuilder()
                .Append("brains:").Append(images.Length);
            int i = 0;
            buffer
                .Append(images[i].layers[0].Synapse.ColumnCount)
                .Append("x")
                .Append(images[i].layers[^1].Synapse.RowCount);
            while (i < combiners.Length)
            {
                buffer.Append(">").Append(combiners[i++].ToString()).Append(">");
                buffer
                    .Append(images[i].layers[0].Synapse.ColumnCount)
                    .Append("x")
                    .Append(images[i].layers[^1].Synapse.RowCount);
            }
            return buffer.ToString();
        }
        public string PrintInfo()
        {
            var buffer = new StringBuilder("[neural network line image]");

            buffer.Append("brains:").Append(images.Length);
            int i = 0;
            buffer.Append("\n")
                .Append(images[i].layers[0].Synapse.ColumnCount)
                .Append("x")
                .Append(images[i].layers[^1].Synapse.RowCount);
            while (i < combiners.Length)
            {
                buffer.Append(">").Append(combiners[i++].ToString()).Append(">");
                buffer.Append("\n")
                    .Append(images[i].layers[0].Synapse.ColumnCount)
                    .Append("x")
                    .Append(images[i].layers[^1].Synapse.RowCount);
            }
            return buffer.ToString();
        }
    }
}
