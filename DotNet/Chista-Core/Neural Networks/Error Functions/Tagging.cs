using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Tagging : IErrorFunction
    {
        public Tagging(double min_accept, double max_reject)
        {
            MinAccept = min_accept;
            MaxReject = max_reject;
        }

        public double MinAccept { get; }
        public double MaxReject { get; }

        public Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> _)
        {
            var error = new double[output.Count];
            for (int i = 0; i < output.Count; i++)
                if (output[i] <= MaxReject) error[i] = output[i] / output.Count;
                else if (output[i] >= MinAccept) error[i] = 1 - output[i] / output.Count;
            return Vector<double>.Build.DenseOfArray(error);
        }
        public double Accuracy(NeuralNetworkFlash flash, double[] _)
        {
            // actually 'TotalError' is avrage of errors
            // becaue in 'NegativeErrorDerivative' function output is divided by count
            return 1 - flash.TotalError;
        }

        public override string ToString()
        {
            return $"Tagging: reject:{MaxReject} accpet:{MinAccept}";
        }
    }
}
