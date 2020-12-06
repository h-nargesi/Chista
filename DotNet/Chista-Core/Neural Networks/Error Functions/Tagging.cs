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

        public Vector<double> ErrorCalculation(Vector<double> output, Vector<double> _)
        {
            var error = new double[output.Count];
            for (int i = 0; i < output.Count; i++)
                if (output[i] >= MinAccept) error[i] = 1 - output[i];
                else if (output[i] <= MaxReject) error[i] = 0 - output[i];
            return Vector<double>.Build.DenseOfArray(error);
        }
        public double Accuracy(NeuralNetworkFlash prediction)
        {
            return (prediction.InputSignals[^1] * 2 - 1).PointwiseAbs().Sum() / prediction.InputSignals[^1].Count;
        }

        public override string ToString()
        {
            return $"Tagging: reject:{MaxReject} accpet:{MinAccept}";
        }
    }
}
