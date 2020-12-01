using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista
{
    class ExtraSignals
    {
        public ExtraSignals(int length)
        {
            Length = length;
            SignalsExtra = new List<Vector<double>[]>();
        }

        public int Length { get; }

        private readonly List<Vector<double>[]> SignalsExtra;
        public Vector<double>[] this[int index]
        {
            get
            {
                while (SignalsExtra.Count <= index)
                    SignalsExtra.Add(new Vector<double>[Length]);
                return SignalsExtra[index];
            }
            set
            {
                while (SignalsExtra.Count <= index)
                    SignalsExtra.Add(new Vector<double>[Length]);
                SignalsExtra[index] = value;
            }
        }

    }
}
