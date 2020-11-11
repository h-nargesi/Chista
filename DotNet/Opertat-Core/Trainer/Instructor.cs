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
        public uint TrainingCount { get; set; }
        public uint ValidationCount { get; set; }
        public uint EvaluationCount { get; set; }
        public uint Offset { get; set; }
        public TraingingStages Stage { get; set; }
        public uint Epoch { get; set; }
        private (uint progress, TraingingStages stage) GetNextRound()
        {
            var progress = Offset + 1;
            switch (Stage)
            {
                case TraingingStages.Training:
                    if (progress >= TrainingCount)
                        return (0, TraingingStages.Validation);
                    else return (progress, Stage);

                case TraingingStages.Validation:
                    if (progress >= ValidationCount)
                        return (0, TraingingStages.Evaluation);
                    else return (progress, Stage);

                case TraingingStages.Evaluation:
                    if (progress >= EvaluationCount)
                        return (0, TraingingStages.Training);
                    else return (progress, Stage);

                default: throw new Exception($"Invalid stage! ({Stage})");
            }
        }
        #endregion


        #region Brain Management

        private readonly List<TrainProcess> processes = new List<TrainProcess>();
        private readonly List<BrainInfo> out_of_line = new List<BrainInfo>();
        public IReadOnlyList<ITrainProcess> Processes => processes;
        public IReadOnlyList<BrainInfo> OutOfLine => out_of_line;
        public void AddProgress(Brain brain)
        {
            lock (processes) processes.Add(new TrainProcess(brain));
        }
        public void RemoveProgress(int index)
        {
            lock (processes) processes.RemoveAt(index);
        }
        public void LoadProgress(
            IReadOnlyList<ITrainProcess> progresses,
            IReadOnlyList<BrainInfo> out_of_line)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            if (progresses == null)
                throw new ArgumentNullException(nameof(progresses));

            this.processes.Clear();
            foreach (var iprg in progresses)
                if (iprg is TrainProcess prg)
                    this.processes.Add(prg);
                else throw new Exception("Invalid progress type.");

            this.out_of_line.Clear();
            if (out_of_line != null)
                foreach (var ibrain in out_of_line)
                    if (ibrain is BrainInfo brn)
                        this.out_of_line.Add(brn);
                    else throw new Exception("Invalid brain-info type.");
        }
        public void AddBrainInfo(BrainInfo brain)
        {
            lock (processes) out_of_line.Add(brain);
        }
        public void RemoveBrainInfo(int index)
        {
            lock (processes) out_of_line.RemoveAt(index);
        }
        #endregion


        #region Abstract Stuff
        protected abstract void OnInitialize();
        protected abstract Task<Record> PrepareNextData(uint progress, TraingingStages stage);
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
                locker.AcquireWriterLock(4096);
                Task<Record> record_geter = null;

                try
                {
                    // initialize by developer
                    OnInitialize();

                    // fetch next record
                    record_geter = PrepareNextData(Offset, Stage);

                    // training loop
                    while (!Canceling && processes.Count > 0)
                    {
                        // current record
                        record_geter.Wait();
                        var record = record_geter.Result;

                        // fetch next record
                        var (offset, stage) = GetNextRound();
                        record_geter = PrepareNextData(offset, stage);

                        if (record != null && record.data != null && record.result != null)
                        {
                            // reporting vriables
                            var start_time = DateTime.Now.Ticks;

                            if (Canceling) break;

                            lock (processes)
                                switch (Stage)
                                {
                                    case TraingingStages.Training:
                                        Parallel.ForEach(processes, (process, state, index) =>
                                        {
                                            // for sure
                                            if (process.OutOfLine) return;
                                            // train this neural network
                                            var flash = process.Brain.Train(record.data, record.result);
                                            // change progress state
                                            process.ChangeSatate(flash);
                                        });

                                        // if next round is first round of stage
                                        // if this round is end round of stage
                                        if (offset == 0)
                                            foreach (var progress in processes)
                                                progress.FinishCurrentState(true);

                                        break;
                                    case TraingingStages.Validation:
                                        Parallel.ForEach(processes, (process, state, index) =>
                                        {
                                            // for sure
                                            if (process.OutOfLine) return;
                                            // test this neural network with validation data
                                            var flash = process.Brain.Test(record.data);
                                            // calculate total error
                                            process.Brain.FillTotalError(flash, record.result);
                                            // change progress state
                                            process.ChangeSatate(flash);
                                        });

                                        // if next round is first round of stage
                                        // if this round is end round of stage
                                        if (offset == 0)
                                            foreach (var progress in processes)
                                                progress.FinishCurrentState(false);

                                        break;
                                    case TraingingStages.Evaluation:
                                        Parallel.ForEach(processes, (process, state, index) =>
                                        {
                                            // do jus out-of-line processes
                                            if (!process.OutOfLine) return;
                                            // test this neural network with evaluation data
                                            var flash = process.Brain.Test(record.data);
                                            // calculate total error
                                            process.Brain.FillTotalError(flash, record.result);
                                            // change progress state
                                            process.ChangeSatate(flash);
                                        });

                                        // if next round is first round of stage
                                        // if this round is end round of stage
                                        if (offset == 0)
                                            for (int p = 0; p < processes.Count;)
                                                if (!processes[p].OutOfLine) p++;
                                                else
                                                {
                                                    out_of_line.Add(new BrainInfo(
                                                        processes[p].BestBrainImage,
                                                        processes[p].BestBrainAccuracy));
                                                    processes.RemoveAt(p);
                                                }

                                        break;
                                }

                            if (Canceling) break;
                            // call event
                            ReflectFinished(record, DateTime.Now.Ticks - start_time);
                        }

                        // next offset
                        if (offset == 0 && stage == TraingingStages.Training)
                            Epoch++;
                        Offset = offset;
                        Stage = stage;
                    }
                }
                catch (Exception ex) { OnError(ex); }
                finally { locker.ReleaseWriterLock(); }
            });
        }
        public void Dispose()
        {
            Canceling = true;
            locker.AcquireWriterLock(4096);
            try { OnStopped(); }
            finally { locker.ReleaseWriterLock(); }
        }
        #endregion


        protected class Record
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

        public static string GetDurationString(long duration)
        {
            var result = new StringBuilder();
            // 100-nanosecond
            result.Insert(0, duration % 10000);
            // millisecond
            duration /= 10000;
            if (duration == 0) return result.ToString();
            result.Insert(0, "ms,").Insert(0, duration % 1000);
            // second
            duration /= 1000;
            if (duration == 0) return result.ToString();
            result.Insert(0, "s,").Insert(0, duration % 60);
            // miniute
            duration /= 60;
            if (duration == 0) return result.ToString();
            result.Insert(0, "m,").Insert(0, duration % 60);
            // hour
            duration /= 60;
            if (duration == 0) return result.ToString();
            result.Insert(0, "h,").Insert(0, duration % 24);
            // days
            duration /= 24;
            if (duration == 0) return result.ToString();
            result.Insert(0, "d,").Insert(0, duration);
            // return
            return result.ToString();
        }
    }
}