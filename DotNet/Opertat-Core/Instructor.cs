using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Photon.NeuralNetwork.Opertat
{
    public abstract class Instructor : IDisposable
    {
        private readonly List<Brain> brains = new List<Brain>();
        private readonly object accuracy_lock = new object();
        private bool stoping = false;

        public uint Offset { get; set; }
        public uint Count { get; set; }
        public uint Epoch { get; set; }
        public uint Tries { get; set; }
        public double Accuracy { get; private set; }
        public IReadOnlyList<Brain> Brains => brains;
        public void BrainAdd(Brain brain)
        {
            lock (brains)
                brains.Add(brain);
        }
        public void BrainRemove(Brain brain)
        {
            lock (brains)
                brains.Remove(brain);
        }
        public void BrainRemove(int index)
        {
            lock (brains)
                brains.RemoveAt(index);
        }

        protected abstract void OnInitialize();
        protected abstract Task<Record> PrepareNextData(uint offset);
        protected abstract void OnError(Exception ex);

        public Task Start()
        {
            return Task.Run(() =>
            {
                try
                {
                    // initialize by developer
                    OnInitialize();
                    // variables
                    var accuracy_total = 0D;
                    var record_count = 0;
                    var record_geter = PrepareNextData(Offset % Count);
                    var start_time = 0L;
                    // training loop
                    while (Offset / Count <= Epoch)
                    {
                        if (stoping) return;

                        // current record
                        record_geter.Wait();
                        var record = record_geter.Result;

                        // fetch next record
                        record_geter = PrepareNextData((Offset + 1) % Count);

                        if (Offset % Count == 0)
                        {
                            accuracy_total = 0;
                            record_count = 0;
                        }
                        double accuracy_last = 0;

                        if (record != null && record.data != null && record.result != null)
                        {
                            int t = 0;
                            Task[] tasks = new Task[brains.Count];

                            lock (brains)
                                foreach (Brain brain in brains)
                                {
                                    tasks[t] = Task.Run(() =>
                                    {
                                        double error;
                                        var i = 1;
                                        do
                                        {
                                            if (stoping) return;
                                            if (ReflectFinished != null) start_time = DateTime.Now.Ticks;

                                            // training
                                            var predict = brain.Train(record.data, record.result);

                                            // check accuracy and error
                                            lock (accuracy_lock)
                                            {
                                                accuracy_last = predict.Accuracy;
                                                error = 1 - accuracy_last;
                                                Accuracy =
                                                    (accuracy_total + accuracy_last) /
                                                    (record_count + 1);
                                            }

                                            // call event
                                            if (ReflectFinished != null)
                                                if (stoping) return;
                                                else ReflectFinished.Invoke(
                                                    t, predict, record, DateTime.Now.Ticks - start_time);
                                        }
                                        while (++i < Tries && error != 0);
                                    });
                                    t++;
                                }
                            Task.WaitAll(tasks);

                            record_count++;
                            accuracy_total += accuracy_last;
                        }

                        // next offset
                        Offset++;
                    }
                    Dispose();
                }
                catch (Exception ex) { OnError(ex); }
            });
        }

        public virtual void Dispose()
        {
            stoping = true;
        }

        protected Action<int, NeuralNetworkFlash, Record, long> ReflectFinished { get; set; }

        protected class Record
        {
            public readonly double[] data, result;
            public readonly object extra;
            public readonly long? duration;

            public Record(double[] data, double[] result, object extra = null, long? duration = null)
            {
                this.data = data;
                this.result = result;
                this.extra = extra;
                this.duration = duration;
            }
        }

        public static string GetDurationString(long duration)
        {
            var result = new StringBuilder();
            // 100-nanosecond
            result.Insert(0, duration % 10000);
            // millisecond
            duration /= 10000;
            if (duration == 0) return result.ToString();
            result.Insert(0, "ms").Insert(0, duration % 1000);
            // second
            duration /= 1000;
            if (duration == 0) return result.ToString();
            result.Insert(0, "s").Insert(0, duration % 60);
            // miniute
            duration /= 60;
            if (duration == 0) return result.ToString();
            result.Insert(0, "m").Insert(0, duration % 60);
            // hour
            duration /= 60;
            if (duration == 0) return result.ToString();
            result.Insert(0, "h").Insert(0, duration % 24);
            // days
            duration /= 24;
            if (duration == 0) return result.ToString();
            result.Insert(0, "d").Insert(0, duration);
            // return
            return result.ToString();
        }
    }
}