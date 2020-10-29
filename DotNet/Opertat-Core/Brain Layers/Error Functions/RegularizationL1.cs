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
            return Matrix<double>.Build.DenseOfArray(new double[synapse.RowCount, synapse.ColumnCount]) + 1;
        }

        public override string ToString()
        {
            return "L1";
        }
    }
}
