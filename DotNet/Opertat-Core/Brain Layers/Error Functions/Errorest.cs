using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class Errorest : IErrorFunction
    {
        public Vector<double> ErrorCalculation(Vector<double> output, Vector<double> values)
        {
            // TODO: use wight to loose certainty
            // error equals to: (true_value - network_output)
            return values - output;
        }

        public override string ToString()
        {
            return "Errorest";
        }
    }
}