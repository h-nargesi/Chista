using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Classification : IErrorFunction
    {
        public Classification(double min_accept)
        {
            MinAccept = min_accept;
        }

        public double MinAccept { get; }

        public Vector<double> ErrorCalculation(Vector<double> output, Vector<double> _)
        {
            var error = new double[output.Count];
            var max_value = 0D;
            var max_index = -1;
            for (int i = 0; i < output.Count; i++)
                if (output[i] >= MinAccept && output[i] > max_value)
                {
                    max_index = i;
                    max_value = output[i];
                }

            for (int i = 0; i < output.Count; i++)
                if (i == max_index) error[max_index] = 1 - output[max_index];
                else error[max_index] = -output[max_index];

            return Vector<double>.Build.DenseOfArray(error);
        }
        public double Accuracy(NeuralNetworkFlash prediction)
        {
            return (prediction.InputSignals[^1] * 2 - 1).PointwiseAbs().Sum() / prediction.InputSignals[^1].Count;
        }

        public override string ToString()
        {
            return $"Classification: accpet:{MinAccept}";
        }
    }
}
