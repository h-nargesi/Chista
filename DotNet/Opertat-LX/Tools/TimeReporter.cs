using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Opertat.Debug.Tools
{
    class TimeReporter
    {
        private readonly LinkedList<long> history =
            new LinkedList<long>();
        private uint max_history_count = 100;
        private long last_printing_time = DateTime.Now.Ticks;

        public uint MaxHistory
        {
            get { return max_history_count; }
            set { max_history_count = value < 1 ? 1 : value; }
        }

        public long GetNextAvg()
        {
            var point = DateTime.Now.Ticks;
            var value = point - last_printing_time;
            last_printing_time = point;

            history.AddLast(value);
            while (history.Count > MaxHistory)
                history.RemoveFirst();

            value = 0;
            foreach (var v in history) value += v;
            return value / history.Count;
        }
    }
}
