using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class Sigmoind : IConduction
    {
        public int ExtraCount => 0;
        public double CertaintyFactor { get; set; }
        public Matrix<double> Regularize(Matrix<double> synapse)
        {
            return synapse * 2 * CertaintyFactor;
        }
        public Vector<double> Conduct(Vector<double> signal)
        {
            return 1 / (1 + (signal * -1).PointwiseExp());
        }
        public Vector<double> Conduct(NeuralNetworkFlash flash, int layer)
        {
            return 1 / (1 + (flash.SignalsSum[layer] * -1).PointwiseExp());
        }
        public Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer)
        {
            return flash.InputSignals[layer + 1]
                .PointwiseMultiply(1 - flash.InputSignals[layer + 1]);
        }
    }
}