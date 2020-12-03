using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class NetProcessImage: INeuralNetworkInformation
    {
        public NetProcessImage(INeuralNetworkImage image, double accuracy)
        {
            Image = image ?? throw new ArgumentNullException(nameof(image));
            Accuracy = accuracy;
        }

        public INeuralNetworkImage Image { get; }
        public double Accuracy { get; }

        public string PrintInfo()
        {
            return $"{Image.PrintInfo()}\naccuracy: {Accuracy}";
        }
    }
}
