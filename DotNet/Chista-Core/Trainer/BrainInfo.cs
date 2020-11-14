using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class BrainInfo
    {
        public BrainInfo(NeuralNetworkImage image, double accuracy)
        {
            this.image = image;
            this.accuracy = accuracy;
        }

        public readonly double accuracy;
        public readonly NeuralNetworkImage image;
    }
}
