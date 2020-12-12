using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class SoftMax : IConduction
    {        
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
            var signal = flash.InputSignals[layer + 1];
            var derivaite = Matrix<double>.Build.DenseOfArray(new double[signal.Count, signal.Count]);

            for (int i = signal.Count - 1; i >= 0; i--)
                for (int j = i; j >= 0; j--)
                    if (i == j) derivaite[i, i] = signal[i] * (1 - signal[i]);
                    else derivaite[i, j] = derivaite[j, i] = -signal[j] * signal[i];

            return derivaite * signal;
        }

        public override string ToString()
        {
            return "SoftMax";
        }
    }
}