using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IDataConvertor
    {
        Vector<double> Standardize(Vector<double> values);
        Vector<double> Normalize(Vector<double> values);
    }
}