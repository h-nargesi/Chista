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
            object extra = null, long? duration = null)
        {
            this.data = data;
            this.result = result;
            this.extra = extra;
            this.duration = duration;
        }
    }

}
