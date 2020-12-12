using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Serializer;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IErrorFunction : ISerializableFunction
    {
        Vector<double> NegativeErrorDerivative(Vector<double> output, Vector<double> values);
        double Accuracy(NeuralNetworkFlash flash);
    }
}