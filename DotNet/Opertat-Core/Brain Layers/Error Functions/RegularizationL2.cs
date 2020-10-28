using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class RegularizationL2 : IRegularization
    {
        public Matrix<double> Regularize(Matrix<double> synapse, double certainty)
        {
            return synapse * certainty;
        }

        public override string ToString()
        {
            return "L2";
        }
    }
}
