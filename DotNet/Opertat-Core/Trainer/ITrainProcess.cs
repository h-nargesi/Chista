using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Opertat.Trainer
{
    public interface ITrainProcess
    {
        public Brain Brain { get; }
        public double CurrentAccuracy { get; }
        public NeuralNetworkFlash LastPredict { get; }
        public NeuralNetworkImage BestBrainImage { get; }
        public double BestBrainAccuracy { get; }
    }
}
