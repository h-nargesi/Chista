using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface INetProcess : INeuralNetworkInformation
    {
        public Brain Brain { get; }
        public double Accuracy { get; }
        public NeuralNetworkFlash LastPrediction { get; }
        public NetProcessInfo ProcessInfo();
    }
}
