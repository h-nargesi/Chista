using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Serializer;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface IAccurateGauge : ISerializableFunction
    {
        public double Accuracy(NeuralNetworkFlash prediction);
    }
}
