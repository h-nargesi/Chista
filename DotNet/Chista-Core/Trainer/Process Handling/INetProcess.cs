using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface INetProcess : INeuralNetworkInformation
    {
        public INeuralNetworkImage StableImage { get; }
        public double StableAccuracy { get; }

        public IChistaNet RunningChistaNet { get; }
        public double RunningAccuracy { get; }
        public NeuralNetworkFlash LastPrediction { get; }

        public NetProcessInfo ProcessInfo();
    }
}
