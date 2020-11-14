using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class SoftReLU : IConduction
    {
        public int ExtraCount => 2;
        public Vector<double> Conduct(Vector<double> signal)
        {
            return (signal.PointwiseExp() + 1).PointwiseLog();
        }
        public Vector<double> Conduct(NeuralNetworkFlash flash, int layer)
        {
            flash[EXP][layer] = flash.SignalsSum[layer].PointwiseMinimum(700).PointwiseExp();
            flash[EXP_1][layer] = flash[EXP][layer] + 1;
            return flash[EXP_1][layer].PointwiseLog();
        }
        public Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer)
        {
            return flash[EXP][layer] / flash[EXP_1][layer];
        }

        private const int EXP = 0;
        private const int EXP_1 = 1;

        public override string ToString()
        {
            return "SoftReLU";
        }
    }
}