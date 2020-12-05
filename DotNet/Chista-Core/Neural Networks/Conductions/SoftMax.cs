using System;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class SoftMax : IConduction
    {
        public int ExtraCount => 0;
        public Vector<double> Conduct(Vector<double> signal)
        {
            signal = signal.PointwiseExp();
#if NAN_CHEK
            ChistaNet.NanTest(signal);
            ChistaNet.NanTest(signal / signal.Sum());
#endif
            return signal / signal.Sum();
        }
        public Vector<double> Conduct(NeuralNetworkFlash flash, int layer)
        {
            var signal = flash.SignalsSum[layer].PointwiseExp();
#if NAN_CHEK
            ChistaNet.NanTest(signal);
            ChistaNet.NanTest(signal / signal.Sum());
#endif
            return signal / signal.Sum();
        }
        public Vector<double> ConductDerivative(NeuralNetworkFlash flash, int layer)
        {
            return flash.InputSignals[layer + 1]
                .PointwiseMultiply(1 - flash.InputSignals[layer + 1]);
        }

        public override string ToString()
        {
            return "SoftMax";
        }
    }
}