using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista
{
    public class NeuralNetworkFlash : INeuralNetworkInformation
    {
        public NeuralNetworkFlash(int size)
        {
            SignalsSum = new Vector<double>[size];
            InputSignals = new Vector<double>[size + 1];
            SignalsExtra = new ExtraSignals(size);
        }

        #region Forward Propagation
        internal readonly Vector<double>[] SignalsSum;
        internal readonly Vector<double>[] InputSignals;
        internal readonly ExtraSignals SignalsExtra;
        public Vector<double> OutputSignal => InputSignals[^1];
        #endregion

        #region Final Stage
        public double[] ResultSignals { get; internal set; }
        internal void SetErrors(Vector<double> errors)
        {
            Errors = errors.ToArray();
            TotalError = errors.PointwiseAbs().Sum();
            ErrorAverage = errors.Count > 0 ? TotalError / errors.Count : 0;
        }
        public double[] Errors { get; private set; }
        public double TotalError { get; private set; }
        public double ErrorAverage { get; private set; }
        public double Accuracy { get; internal set; }
        #endregion

        public override string ToString()
        {
            return @$"average:{ErrorAverage}, total:{TotalError}, count:{ResultSignals.Length}";
        }
        public string PrintInfo()
        {
            return @$"[neural network flash]
error average: {ErrorAverage}, totla error: {TotalError}, count: {ResultSignals.Length}";
        }
    }
}
