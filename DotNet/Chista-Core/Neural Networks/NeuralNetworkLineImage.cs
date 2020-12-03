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
            var buffer = new StringBuilder();
            if (layers != null)
            {
                buffer.Append("layers:").Append(layers.Length);
                if (layers.Length > 0)
                {
                    buffer.Append(layers[0].Synapse.ColumnCount);
                    foreach (var l in layers)
                        buffer.Append("x").Append(l.Synapse.RowCount);
                }
            }
            return buffer.ToString();
        }
        public string PrintInfo()
        {
            var buffer = new StringBuilder("[neural network image]");

            if (input_convertor != null)
                buffer.Append("\n")
                    .Append("input data convertor: ").Append(input_convertor.ToString());

            if (layers != null)
            {
                buffer.Append("\n").Append("layers: ").Append(layers.Length);
                if (layers.Length > 0)
                {
                    buffer.Append("\n\t")
                        .Append("input: ").Append(layers[0].Synapse.ColumnCount).Append(" node(s)");
                    foreach (var l in layers)
                        buffer.Append("\n\t").Append("layer: ")
                            .Append(l.Synapse.RowCount).Append(" node(s)")
                            .Append(" func=").Append(l.Conduction.ToString());
                }
            }

            if (output_convertor != null)
                buffer.Append("\n")
                    .Append("output data convertor: ").Append(output_convertor.ToString());

            if (error_fnc != null)
                buffer.Append("\n")
                    .Append("error function: ").Append(error_fnc.ToString());

            if (regularization != null)
                buffer.Append("\n")
                    .Append("regularization: ").Append(regularization.ToString());

            return buffer.ToString();
        }
    }
}
