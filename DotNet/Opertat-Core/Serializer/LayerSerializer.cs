using System;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    static class LayerSerializer
    {
        public static void Serialize(FileStream stream, Layer[] layers)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");

            byte[] buffer;

            // serialize leyar count
            buffer = BitConverter.GetBytes(layers.Length + 1); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize leyar's neuron count
            foreach (var layer in layers)
            {
                buffer = BitConverter.GetBytes(layer.Synapse.ColumnCount); // 4-bytes
                stream.Write(buffer, 0, buffer.Length);
            }
            buffer = BitConverter.GetBytes(layers[^1].Synapse.RowCount); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize leyar's weights
            for (int l = 0; l < layers.Length; l++)
                for (var i = 0; i < layers[l].Synapse.RowCount; i++)
                {
                    buffer = BitConverter.GetBytes(layers[l].Bias[i]); // 8-bytes
                    stream.Write(buffer, 0, buffer.Length);
                    for (var j = 0; j < layers[l].Synapse.ColumnCount; j++)
                    {
                        buffer = BitConverter.GetBytes(layers[l].Synapse[i, j]); // 8-bytes
                        stream.Write(buffer, 0, buffer.Length);
                    }

                    // serialize conduction function
                    buffer = BitConverter.GetBytes(
                        FunctionSerializer.EnCodeIConduction(layers[l].Conduction)); // 2-bytes
                    stream.Write(buffer, 0, buffer.Length);
                }
        }

        public static Layer[] Restore(FileStream stream, ushort version)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");

            return version switch
            {
                NeuralNetworkSerializer.VERSION => Restore(stream),
                _ => throw new Exception("This version of nni is not supported"),
            };
        }
        private static Layer[] Restore(FileStream stream)
        {
            var buffer_short = new byte[2];
            var buffer_int = new byte[4];
            var buffer_long = new byte[8];

            stream.Read(buffer_int, 0, buffer_int.Length);
            var layer_size = new int[BitConverter.ToInt32(buffer_int, 0)];

            int i;
            for (i = 0; i < layer_size.Length; i++)
            {
                stream.Read(buffer_int, 0, buffer_int.Length);
                layer_size[i] = BitConverter.ToInt32(buffer_int, 0);
            }

            var layers = new Layer[layer_size.Length - 1];
            for (var l = 0; l < layers.Length; l++)
            {
                var synapsees = new double[layer_size[l + 1], layer_size[l]];
                var bias = new double[layer_size[l + 1]];
                IConduction conduction = null;

                for (i = 0; i < bias.Length; i++)
                {
                    stream.Read(buffer_long, 0, buffer_long.Length);
                    bias[i] = BitConverter.ToDouble(buffer_long, 0);
                    for (var j = 0; j < synapsees.GetLength(1); j++)
                    {
                        stream.Read(buffer_long, 0, buffer_long.Length);
                        synapsees[i, j] = BitConverter.ToDouble(buffer_long, 0);
                    }

                    stream.Read(buffer_short, 0, buffer_short.Length);
                    conduction = FunctionSerializer.DecodeIConduction(
                        BitConverter.ToUInt16(buffer_short, 0));
                }

                layers[l] = new Layer(conduction)
                {
                    Synapse = Matrix<double>.Build.DenseOfArray(synapsees),
                    Bias = Vector<double>.Build.DenseOfArray(bias),
                };
            }

            return layers;
        }

    }
}