using System;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Opertat.Implement
{
    public class Layer
    {
        public Matrix<double> Synapse { get; set; }
        public Vector<double> Bias { get; set; }
        public IConduction Conduction { get; }

        public Layer(IConduction conduction)
        {
            Conduction = conduction;
        }

        public Layer Clone()
        {
            return new Layer(Conduction)
            {
                Synapse = Synapse,
                Bias = Bias
            };
        }
    }
}