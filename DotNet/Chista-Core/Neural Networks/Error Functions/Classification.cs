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
            if (max_index >= 0) error[max_index] = 1 - output[max_index];
            output = Vector<double>.Build.DenseOfArray(error);

            return output;
        }
        public double Accuracy(NeuralNetworkFlash prediction)
        {
            return 1 - prediction.ErrorAverage;
        }

        public override string ToString()
        {
            return $"Classification: accpet>{MinAccept}";
        }
    }
}
