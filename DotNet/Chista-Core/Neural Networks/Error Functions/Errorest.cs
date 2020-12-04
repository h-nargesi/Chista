using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Errorest : IErrorFunction
    {
        public Vector<double> ErrorCalculation(Vector<double> output, Vector<double> values)
        {
            // TODO: use wight to loose certainty
            // error equals to: (true_value - network_output)
            return values - output;
        }
        public double Accuracy(NeuralNetworkFlash prediction)
        {
            return 1 - prediction.ErrorAverage;
        }

        public override string ToString()
        {
            return "Errorest";
        }
    }
}