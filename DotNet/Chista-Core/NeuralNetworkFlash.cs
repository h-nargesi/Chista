using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista
{
    public class NeuralNetworkFlash
    {
        internal readonly Vector<double>[] SignalsSum;
        internal readonly Vector<double>[] InputSignals;
        private readonly List<Vector<double>[]> SignalsExtra;
        public double[] ResultSignals { get; internal set; }

        public NeuralNetworkFlash(int size)
        {
            SignalsSum = new Vector<double>[size];
            InputSignals = new Vector<double>[size + 1];
            SignalsExtra = new List<Vector<double>[]>();
        }

        public Vector<double>[] this[int index]
        {
            get
            {
                while (SignalsExtra.Count <= index)
                    SignalsExtra.Add(new Vector<double>[SignalsSum.Length]);
                return SignalsExtra[index];
            }
            set
            {
                while (SignalsExtra.Count <= index)
                    SignalsExtra.Add(new Vector<double>[SignalsSum.Length]);
                SignalsExtra[index] = value;
            }
        }

        public double TotalError { get; internal set; }
        public double Accuracy
        {
            get
            {
                if (ResultSignals == null || ResultSignals.Length < 1) return 0;
                else return 1 - TotalError / ResultSignals.Length;
            }
        }
        public double ErrorAverage
        {
            get
            {
                if (ResultSignals == null || ResultSignals.Length < 1) return 0;
                else return TotalError / ResultSignals.Length;
            }
        }
    }
}
