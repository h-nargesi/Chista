using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class ReLU : IConduction
    {
        public int ExtraCount => 0;
        public double CertaintyFactor { get; set; }
        public Matrix<double> Regularize(Matrix<double> synapse)
        {
            return synapse * 0;
        }
        public Vector<double> Conduct(Vector<double> signal)
        {
            return signal.PointwiseMaximum(0);
        }
        public Vector<double> Conduct(NeuralNetworkFlash flash, int layer)
        {
            return flash.SignalsSum[layer].PointwiseMaximum(0);
        }
        public Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer)
        {
            return flash.InputSignals[layer + 1].PointwiseSign();
        }
    }
}