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

        public Vector<double> ErrorCalculation(NeuralNetworkFlash prediction, Vector<double> _)
        {
            // TODO: implement

            var output = prediction.InputSignals[^1];
            var error = new double[output.Count];
            for (int i = 0; i < output.Count; i++)
                if (output[i] >= MinAccept) error[i] = 1 - output[i];
                else if (output[i] <= MaxReject) error[i] = 0 - output[i];
            output = Vector<double>.Build.DenseOfArray(error);
            prediction.SetErrors(output);
            // TODO: change this
            prediction.Accuracy = 1 - prediction.ErrorAverage;
            return output;
        }

        public override string ToString()
        {
            return $"Tagging: reject<{MaxReject} accpet>{MinAccept}";
        }
    }
}
