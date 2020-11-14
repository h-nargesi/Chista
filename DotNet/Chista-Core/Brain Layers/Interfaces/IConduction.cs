using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IConduction
    {
        int ExtraCount { get; }
        Vector<double> Conduct(Vector<double> signal);
        Vector<double> Conduct(NeuralNetworkFlash flash, int layer);
        Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer);
    }
}