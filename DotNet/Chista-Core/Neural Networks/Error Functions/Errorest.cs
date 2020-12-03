using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Errorest : IErrorFunction
    {
        public Vector<double> ErrorCalculation(NeuralNetworkFlash prediction, Vector<double> values)
        {
            // TODO: use wight to loose certainty
            // error equals to: (true_value - network_output)
            var delta = values - prediction.InputSignals[^1];
            prediction.SetErrors(delta);
            prediction.Accuracy = 1 - prediction.ErrorAverage;
            return delta;
        }

        public override string ToString()
        {
            return "Errorest";
        }
    }
}