using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface ITrainingProcess : INetProcess, INeuralNetworkInformation
    {
        public NeuralNetworkImage BestBrainImage { get; }
        public double BestBrainAccuracy { get; }
    }
}
