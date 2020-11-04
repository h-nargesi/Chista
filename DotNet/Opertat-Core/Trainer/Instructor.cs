using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Photon.NeuralNetwork.Opertat.Trainer
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
        private readonly List<BrainInfo> out_of_line = new List<BrainInfo>();
        public IReadOnlyList<IProgress> Progresses => progresses;
        public IReadOnlyList<BrainInfo> OutOfLine => out_of_line;
        public void BrainAdd(Brain brain)
        {
            lock (progresses) progresses.Add(new Progress(brain));
        }
        public void BrainRemove(int index)
        {
            lock (progresses) progresses.RemoveAt(index);
        }
        public void LoadProgress(
            IReadOnlyList<IProgress> progresses,
            IReadOnlyList<BrainInfo> out_of_line)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            if (progresses == null)
                throw new ArgumentNullException(nameof(progresses));

            this.progresses.Clear();
            foreach (var iprg in progresses)
                if (iprg is Progress prg)
                    this.progresses.Add(prg);
                else throw new Exception("Invalid progress type.");

            this.out_of_line.Clear();
            if (out_of_line != null)
                foreach (var ibrain in out_of_line)
                    if (ibrain is BrainInfo brn)
                        this.out_of_line.Add(brn);
                    else throw new Exception("Invalid brain-info type.");
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
                                {
                                    var to_out_of_line = new List<Progress>();

                                    foreach (var progress in progresses)
                                    {
                                        var finished = progress.FinishCurrentState(!record.training);
                                        if (finished) to_out_of_line.Add(progress);
                                    }

                                    foreach (var pr in to_out_of_line)
                                    {
                                        progresses.Remove(pr);
                                        out_of_line.Add(new BrainInfo(pr.BestBrainImage, pr.BestBrainAccuracy));
                                    }
                                }
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