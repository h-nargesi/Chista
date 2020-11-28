using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Serializer;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IDataConvertor : ISerializableFunction
    {
        Vector<double> Standardize(Vector<double> values);
        Vector<double> Normalize(Vector<double> values);
    }
}