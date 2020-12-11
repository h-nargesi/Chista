using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class CrossEntropy : IErrorFunction
    {
        public Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> values)
        {
            return ((1 - values) / (1 - output)) - (values / output);
        }

        public override string ToString()
        {
            return $"CrossEntropy";
        }
    }
}
