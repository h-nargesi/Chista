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
            var values = Vector<double>.Build.DenseOfArray(new double[output.Count]);
            var max_value = 0D;
            var max_index = -1;
            for (int i = 0; i < output.Count; i++)
                if (output[i] > max_value)
                {
                    max_index = i;
                    max_value = output[i];
                }
            if (max_index > -1) values[max_index] = 1;

            return ((1 - values) / (1 - output)) - (values / output);
        }

        public override string ToString()
        {
            return "Classification";
        }
    }
}
