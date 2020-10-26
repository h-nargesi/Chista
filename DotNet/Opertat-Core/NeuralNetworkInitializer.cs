using System;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class NeuralNetworkInitializer
    {
        private int last_layer_input_count;
        private LinkedList<Layer> layers;
        private IErrorFunction error_func;
        private IDataConvertor in_cvrt, out_cvrt;

        private bool absolut_value = false;
        private IContinuousDistribution distribution = new Normal(0, 0.5);

        public NeuralNetworkInitializer SetInputSize(int input_count)
        {
            if (layers != null)
                throw new Exception("The layers input is already set.");
            if (input_count < 1)
                throw new ArgumentOutOfRangeException(
                    nameof(input_count), "The input size must be non-zero and posetive");
            last_layer_input_count = input_count;
            layers = new LinkedList<Layer>();

            return this;
        }
        public NeuralNetworkInitializer SetDistribution(
            IContinuousDistribution distribution, bool absolute = false)
        {
            if (distribution != null)
                this.distribution = distribution;
            absolut_value = absolute;
            return this;
        }
        public NeuralNetworkInitializer AddLayer(
            IConduction conduction, params int[] node_counts)
        {
            if (layers == null)
                throw new Exception("The layers input is not set yet.");
            if (last_layer_input_count < 0)
                throw new Exception("The layers are closed.");

            if (node_counts == null || node_counts.Length == 0) return this;

            foreach (var node_count in node_counts)
            {
                if (node_count < 1)
                    throw new ArgumentOutOfRangeException(nameof(node_count));

                var synapse = Matrix<double>.Build.Random(
                    node_count, last_layer_input_count, distribution);
                var bias = Vector<double>.Build.Random(node_count, distribution);

                if (absolut_value)
                {
                    synapse = synapse.PointwiseAbs();
                    bias = bias.PointwiseAbs();
                }

                layers.AddLast(
                    new Layer(conduction)
                    {
                        Synapse = synapse,
                        Bias = bias
                    });

                last_layer_input_count = node_count;
            }

            return this;
        }
        public NeuralNetworkInitializer SetCorrection(
            IErrorFunction error_func,
            IDataConvertor input_convertot = null,
            IDataConvertor output_convertot = null)
        {
            if (layers == null)
                throw new Exception("The layers input is not set yet.");

            if (layers.Count < 1)
                throw new Exception("The layers output is not set yet.");

            this.error_func = error_func ??
                throw new ArgumentNullException(nameof(error_func),
                    "The error function is undefined.");

            last_layer_input_count = -1;
            in_cvrt = input_convertot;
            out_cvrt = output_convertot;

            return this;
        }

        public NeuralNetworkImage Image()
        {
            return new NeuralNetworkImage(layers.ToArray(), error_func, in_cvrt, out_cvrt);
        }
    }
}