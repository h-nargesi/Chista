using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class BrainLine : IBrain
    {
        public BrainLine(NeuralNetworkLineImage line_image)
        {
            if (line_image == null)
                throw new ArgumentNullException(nameof(line_image), "The nn-line-image is undefined.");

            brains = new Brain[line_image.images.Length];

            int i = 0;
            foreach (var image in line_image.images)
                brains[i++] = new Brain(image);

            combiners = line_image.combiners;
        }

        private int index;
        private readonly Brain[] brains;
        private readonly IDataCombiner[] combiners;

        public IReadOnlyList<Brain> Brains => brains;
        public IReadOnlyList<IDataCombiner> Combiners => combiners;
        public int Index
        {
            get { return index; }
            set
            {
                lock (brains)
                {
                    if (value < 0 || value >= brains.Length)
                        throw new ArgumentOutOfRangeException(nameof(Index));
                    index = value;
                }
            }
        }

        public NeuralNetworkLineImage Image()
        {
            var images = new NeuralNetworkImage[brains.Length];
            for (int b = 0; b < brains.Length; b++)
                images[b] = brains[b].Image();
            return new NeuralNetworkLineImage(images, combiners, index);
        }
        INeuralNetworkImage IBrain.Image()
        {
            return Image();
        }

        private double[] Stimulate(int index, ref double[] inputs)
        {
            int i = 0;
            double[] result = brains[i].Stimulate(inputs);
            while (i <= index)
            {
                inputs = combiners[i++].Combine(result, inputs);
                result = brains[i].Stimulate(inputs);
            }
            return result;
        }
        public double[] Stimulate(double[] inputs)
        {
            lock (brains) return Stimulate(index, ref inputs);
        }
        public NeuralNetworkFlash Test(double[] inputs, double[] values = null)
        {
            lock (brains)
            {
                // get brains output from frist brain to previous brain
                double[] result = index < 1 ? inputs : Stimulate(index - 1, ref inputs);
                // combine the previous brains' output with input
                inputs = combiners[index - 1].Combine(result, inputs);
                // test last brain
                return brains[index].Test(inputs, values);
            }
        }
        public NeuralNetworkFlash Train(double[] inputs, double[] values)
        {
            lock (brains)
            {
                // get brains output from frist brain to previous brain
                double[] result = index < 1 ? inputs : Stimulate(index - 1, ref inputs);
                // combine the previous brains' output with input
                inputs = combiners[index - 1].Combine(result, inputs);
                // test last brain
                return brains[index].Train(inputs, values);
            }
        }
        public void FillTotalError(NeuralNetworkFlash flash, double[] values)
        {
            brains[index].FillTotalError(flash, values);
        }

        public override string ToString()
        {
            /* THIS CODE IS COPEIED FROM 'NeuralNetworkLineImage'.'ToString()' BECAUSE OF PERFORMANCE */
            var buffer = new StringBuilder()
                .Append("brains:").Append(brains.Length);
            int i = 0;
            buffer.Append(brains[i].InputCount).Append("->").Append(brains[i].OutputCount);
            while (i < combiners.Length)
            {
                buffer.Append(">").Append(combiners[i++].ToString()).Append(">");
                buffer.Append(brains[i].InputCount).Append("x").Append(brains[i].OutputCount);
            }
            return buffer.ToString();
        }
        public string PrintInfo()
        {
            /* THIS CODE IS COPEIED FROM 'NeuralNetworkLineImage'.'PrintInfo()' BECAUSE OF PERFORMANCE */
            var buffer = new StringBuilder("[brain line]");

            buffer.Append("brains:").Append(brains.Length);
            int i = 0;
            buffer.Append("\n")
                .Append(brains[i].InputCount).Append("->").Append(brains[i].OutputCount);
            while (i < combiners.Length)
            {
                buffer.Append(">").Append(combiners[i++].ToString()).Append(">");
                buffer.Append("\n")
                    .Append(brains[i].InputCount).Append("x").Append(brains[i].OutputCount);
            }
            return buffer.ToString();
        }
    }
}
