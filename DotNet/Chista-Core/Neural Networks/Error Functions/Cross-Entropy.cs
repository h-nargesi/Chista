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
            return -values / output.PointwiseMaximum(1E-320D);
        }
        public double Accuracy(NeuralNetworkFlash flash)
        {
            var error = Vector<double>.Build.DenseOfArray(flash.Errors);
            return (error.PointwiseSign() - error).PointwiseAbs().Sum() / error.Count;
        }

        public override string ToString()
        {
            return $"CrossEntropy";
        }
    }
}
