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
        #region Progress Control
        public uint Offset { get; set; }
        public uint Count { get; set; }
        public uint Epoch { get; set; }
        public uint Tries { get; set; }
        #endregion


        #region Brain Management

        private readonly List<Progress> progresses = new List<Progress>();
        public IReadOnlyList<IProgress> Progresses => progresses;
        public void BrainAdd(Brain brain)
        {
            lock (progresses) progresses.Add(new Progress(brain));
        }
        public void BrainRemove(int index)
        {
            lock (progresses) progresses.RemoveAt(index);
        }
        #endregion


        #region Abstract Stuff
        protected abstract void OnInitialize();
        protected abstract Task<Record> PrepareNextData(uint offset);
        protected abstract void ReflectFinished(Record record, long duration);
        protected abstract void OnError(Exception ex);
        protected abstract void OnStopped();
        #endregion


        #region Progress Job

        private readonly ReaderWriterLock locker = new ReaderWriterLock();
        public bool Canceling { get; private set; } = false;
        public bool Stopped => locker.IsWriterLockHeld;
        public Task Start()
        {
            return Task.Run(() =>
            {
                Canceling = false;
                locker.AcquireWriterLock(2048);
                Task<Record> record_geter = null;

                try
                {
                    // initialize by developer
                    OnInitialize();

                    // fetch next record
                    record_geter = PrepareNextData(Offset);
                    var prv_was_training = true;

                    // training loop
                    while (!Canceling && Offset / Count <= Epoch)
                    {
                        // current record
                        record_geter.Wait();
                        var record = record_geter.Result;

                        // fetch next record
                        record_geter = PrepareNextData((Offset + 1) % Count);

                        if (record != null && record.data != null && record.result != null)
                        {
                            // reporting vriables
                            var start_time = DateTime.Now.Ticks;

                            if (Canceling) break;

                            lock (progresses)
                            {
                                Parallel.ForEach(progresses, (progress, state, index) =>
                                  {
                                      NeuralNetworkFlash flash = null;

                                      var i = 1;
                                      if (!record.training) flash = progress.Brain.Test(record.data);
                                      else do { flash = progress.Brain.Train(record.data, record.result); }
                                          while (!Canceling && ++i < Tries && flash.Accuracy < 1);

                                      // change progress state
                                      progress.ChangeSatate(flash);
                                  });

                                if (record.training != prv_was_training)
                                    Parallel.ForEach(progresses, (progress, state, index) =>
                                        progress.FinishCurrentState(!record.training));
                            }

                            if (Canceling) break;
                            // call event
                            ReflectFinished(record, DateTime.Now.Ticks - start_time);
                            prv_was_training = record.training;
                        }

                        // next offset
                        Offset++;
                    }
                }
                catch (Exception ex) { OnError(ex); }
                finally { locker.ReleaseWriterLock(); }
            });
        }
        public void Dispose()
        {
            Canceling = true;
            locker.AcquireWriterLock(2048);
            OnStopped();
            locker.ReleaseWriterLock();
        }
        #endregion


        #region Progress History
        private class BrainInfo
        {
            public BrainInfo(NeuralNetworkImage image, double accuracy)
            {
                this.image = image;
                this.accuracy = accuracy;
            }

            public readonly double accuracy;
            public readonly NeuralNetworkImage image;
        }

        private class History
        {
            public History()
            {
                history = new LinkedList<BrainInfo>();
            }

            private readonly LinkedList<BrainInfo> history;
            public BrainInfo BestBrainInfo
            {
                get { return history.First?.Value; }
            }

            public bool AddProgress(IProgress progress)
            {
                if (history.Count < 1 || progress.CurrentAccuracy > history.First.Value.accuracy)
                {
                    history.Clear();
                    history.AddLast(new BrainInfo(progress.Brain.Image(), progress.CurrentAccuracy));
                    return true;
                }
                else
                {
                    history.AddLast(new BrainInfo(null, progress.CurrentAccuracy));

                    double prv_accuracy = 0;
                    int descenting_count = 0;
                    foreach (var info in history)
                    {
                        if (prv_accuracy < info.accuracy) descenting_count = 0;
                        else descenting_count++;
                        prv_accuracy = info.accuracy;
                    }

                    if (descenting_count >= 4) return false;
                    else if (history.Count > 10) return false;
                    else return true;
                }
            }
        }

        public interface IProgress
        {
            public Brain Brain { get; }
            public double CurrentAccuracy { get; }
            public NeuralNetworkFlash LastPredict { get; }
            public NeuralNetworkImage BestBrainImage { get; }
            public double BestBrainAccuracy { get; }
        }

        private class Progress : IProgress
        {
            private readonly History history = new History();
            private int record_count;
            private double total_accuracy;

            public Progress(Brain brain)
            {
                Brain = brain ??
                    throw new ArgumentNullException(nameof(brain), "Instructor.Progress: brain is null");
            }

            public Brain Brain { get; }
            public double CurrentAccuracy { get; private set; }
            public NeuralNetworkFlash LastPredict { get; private set; }

            public void ChangeSatate(NeuralNetworkFlash predict)
            {
                record_count++;
                total_accuracy += predict.Accuracy;
                CurrentAccuracy = total_accuracy / record_count;
                LastPredict = predict;
            }
            public bool FinishCurrentState(bool is_validated)
            {
                record_count = 0;
                total_accuracy = 0;
                CurrentAccuracy = 0;

                if (!is_validated) return true;
                else return history.AddProgress(this);
            }

            public NeuralNetworkImage BestBrainImage
            {
                get { return history.BestBrainInfo.image; }
            }
            public double BestBrainAccuracy
            {
                get { return history.BestBrainInfo.accuracy; }
            }
        }
        #endregion


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