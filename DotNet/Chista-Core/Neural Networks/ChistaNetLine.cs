using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class ChistaNetLine : IChistaNet
    {
        public ChistaNetLine(NeuralNetworkLineImage line_image)
        {
            if (line_image == null)
                throw new ArgumentNullException(nameof(line_image), "The nn-line-image is undefined.");

            chista_nets = new ChistaNet[line_image.images.Length];

            int i = 0;
            foreach (var image in line_image.images)
                chista_nets[i++] = new ChistaNet(image);

            combiners = line_image.combiners;
        }

        private int index;
        private readonly ChistaNet[] chista_nets;
        private readonly IDataCombiner[] combiners;

        public double LearningFactor
        {
            get { return chista_nets[0].LearningFactor; }
            set { foreach (var cn in chista_nets) cn.LearningFactor = value; }
        }
        public double CertaintyFactor
        {
            get { return chista_nets[0].CertaintyFactor; }
            set { foreach (var cn in chista_nets) cn.CertaintyFactor = value; }
        }
        public double DropoutFactor
        {
            get { return chista_nets[0].DropoutFactor; }
            set { foreach (var cn in chista_nets) cn.DropoutFactor = value; }
        }
        public IReadOnlyList<ChistaNet> ChistaNets => chista_nets;
        public IReadOnlyList<IDataCombiner> Combiners => combiners;
        public int Index
        {
            get { return index; }
            set
            {
                lock (chista_nets)
                {
                    if (value < 0 || value >= chista_nets.Length)
                        throw new ArgumentOutOfRangeException(nameof(Index));
                    index = value;
                }
            }
        }

        public NeuralNetworkLineImage Image()
        {
            var images = new NeuralNetworkImage[chista_nets.Length];
            for (int b = 0; b < chista_nets.Length; b++)
                images[b] = chista_nets[b].Image();
            return new NeuralNetworkLineImage(images, combiners, index);
        }
        INeuralNetworkImage IChistaNet.Image()
        {
            return Image();
        }

        private double[] Stimulate(int index, ref double[] inputs)
        {
            int i = 0;
            double[] result = chista_nets[i].Stimulate(inputs);
            while (i <= index)
            {
                inputs = combiners[i++].Combine(result, inputs);
                result = chista_nets[i].Stimulate(inputs);
            }
            return result;
        }
        public double[] Stimulate(double[] inputs)
        {
            lock (chista_nets) return Stimulate(index, ref inputs);
        }
        public NeuralNetworkFlash Test(double[] inputs, double[] values = null)
        {
            lock (chista_nets)
            {
                if (index > 0)
                {
                    // get chista-net's output from frist net to previous chista-net
                    var result = Stimulate(index - 1, ref inputs);
                    // combine the previous chista-nets' output with input
                    inputs = combiners[index - 1].Combine(result, inputs);
                }
                // test last chista-net
                return chista_nets[index].Test(inputs, values);
            }
        }
        public NeuralNetworkFlash Train(double[] inputs, double[] values)
        {
            lock (chista_nets)
            {
                if (index > 0)
                {
                    // get chista-net's output from frist net to previous chista-net
                    var result = Stimulate(index - 1, ref inputs);
                    // combine the previous chista-nets' output with input
                    inputs = combiners[index - 1].Combine(result, inputs);
                }
                // test last chista-net
                return chista_nets[index].Train(inputs, values);
            }
        }
        public void FillTotalError(NeuralNetworkFlash flash, double[] values)
        {
            chista_nets[index].FillTotalError(flash, values);
        }

        public override string ToString()
        {
            /* THIS CODE IS COPEIED FROM 'NeuralNetworkLineImage'.'ToString()' BECAUSE OF PERFORMANCE */
            var buffer = new StringBuilder()
                .Append("chista-nets:").Append(chista_nets.Length);
            int i = 0;
            buffer.Append(chista_nets[i].InputCount).Append("x").Append(chista_nets[i].OutputCount);
            while (i < combiners.Length)
            {
                buffer.Append(">").Append(combiners[i++].ToString()).Append(">");
                buffer.Append(chista_nets[i].InputCount).Append("x").Append(chista_nets[i].OutputCount);
            }
            return buffer.ToString();
        }
        public string PrintInfo()
        {
            /* THIS CODE IS COPEIED FROM 'NeuralNetworkLineImage'.'PrintInfo()' BECAUSE OF PERFORMANCE */
            var buffer = new StringBuilder("[chista-net line]");

            buffer.Append("nets:").Append(chista_nets.Length);
            int i = 0;
            buffer.Append("\n")
                .Append(chista_nets[i].InputCount).Append("x").Append(chista_nets[i].OutputCount);
            while (i < combiners.Length)
            {
                buffer.Append(">").Append(combiners[i++].ToString()).Append(">");
                buffer.Append("\n")
                    .Append(chista_nets[i].InputCount).Append("x").Append(chista_nets[i].OutputCount);
            }
            return buffer.ToString();
        }
    }
}
