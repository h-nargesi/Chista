using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class SoftMax : IConduction
    {
        private static readonly Dictionary<int, Vector<double>> vector_one =
            new Dictionary<int, Vector<double>>();
        
        public int ExtraCount => 0;
        public Vector<double> Conduct(Vector<double> signal)
        {
            signal = (signal - signal.Maximum()).PointwiseExp();
            return signal / signal.Sum();
        }
        public Vector<double> Conduct(NeuralNetworkFlash flash, int layer)
        {
            var signal = flash.SignalsSum[layer];
            signal = (signal - signal.Maximum()).PointwiseExp();
            return signal / signal.Sum();
        }
        public Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer)
        {
            // count of this layer
            layer = flash.InputSignals[layer + 1].Count;
            // return existing vector
            if (vector_one.ContainsKey(layer)) return vector_one[layer];

            // build a new vector
            var one_list = Vector<double>.Build.DenseOfArray(new double[layer]) + 1;
            vector_one.Add(layer, one_list);
            return one_list;
        }
        public Vector<double> ConductDerivative(Vector<double> delta, NeuralNetworkFlash flash, int layer)
        {
            var signal = flash.InputSignals[layer + 1];
            var derivaite = Matrix<double>.Build.DenseOfArray(new double[signal.Count, signal.Count]);

            for (int i = signal.Count - 1; i >= 0; i--)
                for (int j = i; j >= 0; j--)
                    if (i == j) derivaite[i, i] = signal[i] * (1 - signal[i]);
                    else derivaite[i, j] = derivaite[j, i] = -signal[i] * signal[j];

            return derivaite * delta;
        }

        public override string ToString()
        {
            return "SoftMax";
        }
    }
}