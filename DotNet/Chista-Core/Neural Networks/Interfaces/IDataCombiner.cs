using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Serializer;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public interface IDataCombiner : ISerializableFunction
    {
        public double[] Combine(double[] output, double[] data);
    }
}
