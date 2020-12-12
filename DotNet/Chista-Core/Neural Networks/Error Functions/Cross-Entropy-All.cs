using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class CrossEntropyAll : IErrorFunction
    {
        public Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> values)
        {
            throw new Exception("CrossEntropyAll is not tested.");
            // return (values - output) / (output.PointwiseMaximum(1E-320D) * (1 - output));
        }
        public double Accuracy(NeuralNetworkFlash flash)
        {
            var error = Vector<double>.Build.DenseOfArray(flash.Errors);
            return (error.PointwiseSign() - error).PointwiseAbs().Sum() / error.Count;
        }

        public override string ToString()
        {
            return $"CrossEntropyAll";
        }
    }
}
