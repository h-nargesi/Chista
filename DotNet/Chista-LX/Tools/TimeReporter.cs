using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Debug.Tools
{
    class TimeReporter
    {
        private readonly LinkedList<long> history =
            new LinkedList<long>();
        private uint max_history_count = 1000;
        private long last_printing_time = DateTime.Now.Ticks;
        private long current_sum = 0;

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
            current_sum += value;

            while (history.Count > max_history_count)
            {
                current_sum -= history.First.Value;
                history.RemoveFirst();
            }

            return current_sum / history.Count;
        }
    }
}
