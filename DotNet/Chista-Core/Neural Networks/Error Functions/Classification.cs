using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Classification : IErrorFunction
    {
        public Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> _)
        {
            var max_index = output.MaximumIndex();

            var values = Vector<double>.Build.DenseOfArray(new double[output.Count]);
            if (max_index > -1) values[max_index] = -1 / Math.Max(1E-320D, output[max_index]);

            return values;
        }
        public double Accuracy(NeuralNetworkFlash flash, double[] values)
        {
            return values[flash.InputSignals[^1].MaximumIndex()];
        }

        public override string ToString()
        {
            return "Classification";
        }
    }
}
