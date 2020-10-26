using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Opertat.Implement
{
    public interface IRegularization
    {
        Matrix<double> Regularize(Matrix<double> synapse, double certainty);
    }
}