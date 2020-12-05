using System;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class NeuralNetworkInitializer
    {
        private int last_layer_input_count;
        private LinkedList<Layer> layers;
        private IErrorFunction error_func;
        private IDataConvertor in_cvrt, out_cvrt;
        private IRegularization regularization;

        private bool absolut_value = false;
        private IContinuousDistribution distribution = new Normal(0, 0.5);

        private readonly List<NeuralNetworkImage> images = new List<NeuralNetworkImage>();
        private readonly List<IDataCombiner> combiners = new List<IDataCombiner>();

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
            if (last_layer_input_count < 0)
                throw new Exception("The layers are closed.");

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
            IErrorFunction error_func, IRegularization regularization = null)
        {
            if (layers == null)
                throw new Exception("The layers input is not set yet.");

            if (layers.Count < 1)
                throw new Exception("The layers output is not set yet.");

            this.error_func = error_func ??
                throw new ArgumentNullException(nameof(error_func),
                    "The error function is undefined.");

            last_layer_input_count = -1;
            this.regularization = regularization;

            return this;
        }
        public NeuralNetworkInitializer SetDataConvertor(
            IDataConvertor input_convertot, IDataConvertor output_convertot)
        {
            if (layers == null)
                throw new Exception("The layers input is not set yet.");

            if (layers.Count < 1)
                throw new Exception("The layers output is not set yet.");

            if (last_layer_input_count > -1)
                throw new Exception("The layers are not closed.");

            in_cvrt = input_convertot;
            out_cvrt = output_convertot;

            return this;
        }
        public NeuralNetworkInitializer SetDataCombiner(IDataCombiner combiner)
        {
            if (layers == null)
                throw new Exception("The layers input is not set yet.");

            if (combiner == null) throw new ArgumentNullException(nameof(combiner));
            combiners.Add(combiner);
            images.Add(CloseCurrentImage());

            return this;
        }
        private NeuralNetworkImage CloseCurrentImage()
        {
            var image = new NeuralNetworkImage(
                layers.ToArray(),
                error_func, in_cvrt, out_cvrt, regularization);

            last_layer_input_count = 0;
            layers = null;
            error_func = null;
            in_cvrt = out_cvrt = null;
            regularization = null;

            return image;
        }

        public INeuralNetworkImage Image()
        {
            if (layers != null)
                images.Add(CloseCurrentImage());

            if (images.Count == 1) return images[0];

            else return new NeuralNetworkLineImage(images.ToArray(), combiners.ToArray(), 0);
        }
        public IChistaNet ChistaNet(double learning, double certainty, double dropout)
        {
            var image = Image();

            return image switch
            {
                NeuralNetworkImage simple_image =>
                    new ChistaNet(simple_image)
                    {
                        LearningFactor = learning,
                        CertaintyFactor = certainty,
                        DropoutFactor = dropout
                    },
                NeuralNetworkLineImage line_image =>
                    new ChistaNetLine(line_image)
                    {
                        LearningFactor = learning,
                        CertaintyFactor = certainty,
                        DropoutFactor = dropout
                    },
                _ => throw new Exception("Invalid chista-net type"),
            };
        }
    }
}