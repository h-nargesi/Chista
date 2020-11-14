using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IRegularization
    {
        Matrix<double> Regularize(Matrix<double> synapse, double certainty);
    }
}