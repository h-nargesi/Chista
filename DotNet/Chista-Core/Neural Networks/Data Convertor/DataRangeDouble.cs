using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class DataRangeDouble : IDataConvertor
    {
        public double SignalRange { get; private set; } = 1;
        public double SignalHeight { get; private set; } = 0;

        public DataRangeDouble(double range, double height)
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