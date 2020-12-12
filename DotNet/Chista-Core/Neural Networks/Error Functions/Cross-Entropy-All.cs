using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class CrossEntropyAll : IErrorFunction
    {
        private const double MIN = 1E-320D, MAX = 1 - MIN;

        public Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> values)
        {
            //throw new Exception("CrossEntropyAll is not tested.");
            return -(values - output) / (output.PointwiseMaximum(MIN) * (1 - output.PointwiseMinimum(MAX)));
        }
        public double Accuracy(NeuralNetworkFlash flash, double[] values)
        {
            return values[flash.InputSignals[^1].MaximumIndex()];
        }

        public override string ToString()
        {
            return $"CrossEntropyAll";
        }
    }
}
