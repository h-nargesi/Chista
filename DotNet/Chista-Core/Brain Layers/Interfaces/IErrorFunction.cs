using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Serializer;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IErrorFunction : ISerializableFunction
    {
        Vector<double> ErrorCalculation(NeuralNetworkFlash prediction, Vector<double> values);
    }
}