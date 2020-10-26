using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class NeuralNetworkImage
    {
        public readonly Layer[] layers;
        public readonly IErrorFunction error_fnc;
        public readonly IDataConvertor input_convertor, output_convertor;
        public readonly IRegularization regularization;

        public NeuralNetworkImage(
            Layer[] layers, IErrorFunction error_fnc,
            IDataConvertor input_convertor, IDataConvertor output_convertor,
            IRegularization regularization)
        {
            CheckImageError(layers, error_fnc);
            this.layers = layers;
            this.error_fnc = error_fnc;
            this.input_convertor = input_convertor;
            this.output_convertor = output_convertor;
            this.regularization = regularization;
        }

        public static void CheckImageError(Layer[] layers, IErrorFunction error_fnc)
        {
            if (layers == null)
                throw new ArgumentNullException(nameof(layers),
                    "The nn-image's layers are undefined.");

            if (layers.Length < 1)
                throw new ArgumentException(nameof(layers),
                    "The nn-image's layers are empty.");

            if (error_fnc == null)
                throw new ArgumentNullException(nameof(error_fnc),
                    "The error function is undefined.");

            string messages = null;

            int? prv_length = null;
            var i = 0;
            while (i < layers.Length)
            {
                if (layers[i].Synapse.RowCount < 1)
                    messages += $"\r\nThe number of neuron of layar {i} is zero.";

                if (layers[i].Bias.Count != layers[i].Synapse.RowCount)
                    messages += $"\r\nThe number of neuron's bias of layar {i} is not correct.";

                if (prv_length != null && prv_length != layers[i].Synapse.ColumnCount)
                    messages += $"\r\nThe number of synapsees of layar {i} is not match with previous layer's neuron count.";
                prv_length = layers[i].Synapse.RowCount;

                i++;
            }

            if (messages != null)
                throw new ArgumentException(nameof(layers), messages);
        }

    }
}
