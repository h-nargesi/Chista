using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Photon.NeuralNetwork.Opertat
{
    public abstract class Instructor : IDisposable
    {
        private bool stoping = false;
        private readonly Dictionary<Brain, (double accuracy, NeuralNetworkFlash predict)> brains =
            new Dictionary<Brain, (double accuracy, NeuralNetworkFlash predict)>();

        #region Progress Control
        public uint Offset { get; set; }
        public uint TraingingCount { get; set; }
        public uint ValidationCount { get; set; }
        public uint TotalCount => TraingingCount + ValidationCount;
        public uint Epoch { get; set; }
        public uint Tries { get; set; }
        private (uint, bool) OffsetCurrent(uint next)
        {
            next = (Offset + next) % TotalCount;
            bool is_training_process = next < TraingingCount;
            if (!is_training_process) next -= TraingingCount;

            return (next, is_training_process);
        }
        #endregion

        #region Brain Management
        public IReadOnlyDictionary<Brain, (double accuracy, NeuralNetworkFlash predict)> Brains => brains;
        public void BrainAdd(Brain brain)
        {
            lock (brains) brains.Add(brain, (0, null));
        }
        public void BrainRemove(Brain brain)
        {
            lock (brains) brains.Remove(brain);
        }
        public void BrainRemove(int index)
        {
            lock (brains)
            {
                foreach (Brain brain in brains.Keys)
                    if (index-- == 0)
                    {
                        brains.Remove(brain);
                        break;
                    }
            }
        }
        #endregion

        #region Abstract Stuff
        protected abstract void OnInitialize();
        protected abstract Task<Record> PrepareNextData(uint offset, bool training);
        protected abstract void ReflectFinished(Record record, long duration);
        protected abstract void OnError(Exception ex);
        #endregion

        #region Best Validation
        // TODO: store best validation in memory
        // TODO: restore best validation in dispose
        #endregion

        public Task Start()
        {
            return Task.Run(() =>
            {
                try
                {
                    // initialize by developer
                    OnInitialize();

                    // variables
                    var (offest, is_training) = OffsetCurrent(0);
                    var record_count = 0;
                    var accuracy_total = new Dictionary<Brain, double>();
                    var record_geter = PrepareNextData(offest, is_training);

                    // initial total accuracy for each brain
                    foreach (var brain in brains.Keys)
                        accuracy_total.Add(brain, 0);

                    // training loop
                    while (Offset / TotalCount <= Epoch)
                    {
                        if (stoping) return;

                        // current record
                        record_geter.Wait();
                        var record = record_geter.Result;

                        // fetch next record
                        var (next, next_is_training) = OffsetCurrent(1);
                        record_geter = PrepareNextData(next, next_is_training);

                        if (offest == 0)
                        {
                            record_count = 0;
                            accuracy_total.Clear();
                            foreach (var brain in brains.Keys)
                                accuracy_total.Add(brain, 0);
                        }

                        if (record != null && record.data != null && record.result != null)
                        {
                            // reporting vriables
                            record_count++;
                            var start_time = DateTime.Now.Ticks;

                            lock (brains)
                                _ = Parallel.ForEach(brains.Keys.ToArray(), (brain, state, index) =>
                                  {
                                      NeuralNetworkFlash flash = null;

                                      var i = 1;
                                      if (!record.training) flash = brain.Test(record.data);
                                      else do { flash = brain.Train(record.data, record.result); }
                                      while (!stoping && ++i < Tries && flash.Accuracy < 1);

                                      // total accuracy
                                      accuracy_total[brain] += flash.Accuracy;

                                      // report_brain
                                      brains[brain] = (accuracy_total[brain] / record_count, flash);
                                  });

                            // call event
                            if (stoping) return;
                            ReflectFinished(record, DateTime.Now.Ticks - start_time);
                        }

                        // next offset
                        Offset++;
                        offest = next;
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

        protected class Record
        {
            public readonly bool training;
            public readonly double[] data, result;
            public readonly object extra;
            public readonly long? duration;

            public Record(bool training, double[] data, double[] result, 
                object extra = null, long? duration = null)
            {
                this.training = training;
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