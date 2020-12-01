using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class AccurateGauge : IAccurateGauge
    {
        public double Accuracy(NeuralNetworkFlash prediction)
        {
            return 1 - prediction.ErrorAverage;
        }
    }
}
