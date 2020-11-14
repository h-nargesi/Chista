using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IErrorFunction
    {
        Vector<double> ErrorCalculation(Vector<double> output, Vector<double> values);
    }
}