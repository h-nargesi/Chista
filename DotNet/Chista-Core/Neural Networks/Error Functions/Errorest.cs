using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Errorest : IErrorFunction
    {
        public Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> values)
        {
            // TODO: use wight to loose certainty
            // error equals to: (true_value - network_output)
            return (values - output) / output.Count;
        }

        public override string ToString()
        {
            return "Errorest";
        }
    }
}