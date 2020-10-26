using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Opertat.Implement
{
    public interface IConduction
    {
        int ExtraCount { get; }
        double CertaintyFactor { get; set; }
        Matrix<double> Regularize(Matrix<double> synapse);
        Vector<double> Conduct(Vector<double> signal);
        Vector<double> Conduct(NeuralNetworkFlash flash, int layer);
        Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer);
    }
}