using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class RegularizationL1 : IRegularization
    {
        public Matrix<double> Regularize(Matrix<double> synapse, double certainty)
        {
            return synapse.PointwiseSign() * certainty;
        }

        public override string ToString()
        {
            return "L1";
        }
    }
}
