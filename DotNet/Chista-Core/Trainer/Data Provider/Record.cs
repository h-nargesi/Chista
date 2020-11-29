using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class Record
    {
        public readonly double[] data, result;
        public readonly object extra;
        public readonly long? duration;

        public Record(double[] data, double[] result,
            long? duration = null, object extra = null)
        {
            this.data = data;
            this.result = result;
            this.duration = duration;
            this.extra = extra;
        }
    }

}
