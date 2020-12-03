using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class Straight : IConduction
    {
        Dictionary<int, Vector<double>> vector_one = new Dictionary<int, Vector<double>>();

        public int ExtraCount => 0;
        public Vector<double> Conduct(Vector<double> signal)
        {
            return signal;
        }
        public Vector<double> Conduct(NeuralNetworkFlash flash, int layer)
        {
            return flash.SignalsSum[layer];
        }
        public Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer)
        {
            // count of this layer
            layer = flash.InputSignals[layer + 1].Count;
            // return existing vector
            if (vector_one.ContainsKey(layer)) return vector_one[layer];
            // build a new vector
            var one_list = Vector<double>.Build.DenseOfArray(new double[layer]) + 1;
            vector_one.Add(layer, one_list);
            return one_list;
        }

        public override string ToString()
        {
            return "Stright";
        }
    }
}