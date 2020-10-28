using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class DataRange : IDataConvertor
    {
        public uint SignalRange { get; private set; } = 1;
        public int SignalHeight { get; private set; } = 0;

        public DataRange(uint range, int height)
        {
            if (range < 1)
                throw new ArgumentOutOfRangeException(
                    nameof(range), "The range cannot be zero.");

            SignalRange = range;
            SignalHeight = height;
        }

        public Vector<double> Standardize(Vector<double> values)
        {
            if (SignalHeight != 0) values += SignalHeight;
            if (SignalRange > 1) values /= SignalRange;
            return values;
        }
        public Vector<double> Normalize(Vector<double> values)
        {
            if (SignalRange > 1) values *= SignalRange;
            if (SignalHeight != 0) values -= SignalHeight;
            return values;
        }

        public override string ToString()
        {
            return $"f(data) = (data + {SignalHeight}) / {SignalRange}";
        }
    }
}