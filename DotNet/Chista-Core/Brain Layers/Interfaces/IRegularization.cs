using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Serializer;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IRegularization : ISerializableFunction
    {
        Matrix<double> Regularize(Matrix<double> synapse, double certainty);
    }
}