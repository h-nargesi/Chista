using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface ITrainProcess : INeuralNetworkInformation
    {
        public Brain Brain { get; }
        public double CurrentAccuracy { get; }
        public NeuralNetworkFlash LastPredict { get; }
        public NeuralNetworkImage BestBrainImage { get; }
        public double BestBrainAccuracy { get; }

        public ProgressInfo ProgressInfo();
    }
}
