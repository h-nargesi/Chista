using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Chista.Trainer.Delegates;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public abstract class Instructor
    {
        public Instructor(IDataProvider data_provider)
        {
            this.data_provider = data_provider ?? throw new ArgumentNullException(nameof(data_provider));
        }

        #region Progress Control

        private IDataProvider data_provider;
        public IDataProvider DataProvider
        {
            get { return data_provider; }
            set
            {
                if (value == null) throw new ArgumentNullException(nameof(value));
                else if (!Stopped) throw new Exception(
                    "Can not change the data-provider while instructor is running.");

                Stage = TraingingStages.Training;
                Offset = 0;
                Epoch = 0;

                data_provider = value;
            }
        }
        public TraingingStages Stage { get; set; } = TraingingStages.Training;
        public uint Offset { get; set; }
        public uint Epoch { get; set; }
        public uint EpochMax { get; set; }
        private (uint progress, TraingingStages stage) GetNextRound()
        {
            var progress = Offset + 1;
            switch (Stage)
            {
                case TraingingStages.Training:
                    if (progress >= data_provider.TrainingCount)
                        return (0, TraingingStages.Validation);
                    else return (progress, Stage);

                case TraingingStages.Validation:
                    if (progress >= data_provider.ValidationCount)
                        return (0, TraingingStages.Evaluation);
                    else return (progress, Stage);

                case TraingingStages.Evaluation:
                    if (progress >= data_provider.EvaluationCount)
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
        public void LoadProgress(InstructorProcessInfo process_info)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            if (process_info == null)
                throw new ArgumentNullException(nameof(process_info));

            lock (processes)
            {
                processes.Clear();
                if (process_info.Processes != null)
                    foreach (var iprg in process_info.Processes)
                        if (iprg is TrainProcess prg) processes.Add(prg);
                        else throw new Exception("Invalid progress type.");
            }

            lock (out_of_line)
            {
                out_of_line.Clear();
                if (process_info.OutOfLine != null)
                    foreach (var ibrain in process_info.OutOfLine)
                        if (ibrain is BrainInfo brn) out_of_line.Add(brn);
                        else throw new Exception("Invalid brain-info type.");
            }

            Epoch = process_info.Epoch;
            Stage = process_info.Stage;
            Offset = process_info.Offset;
        }
        public void AddBrainInfo(BrainInfo brain)
        {
            lock (out_of_line) out_of_line.Add(brain);
        }
        public void RemoveBrainInfo(int index)
        {
            lock (out_of_line) out_of_line.RemoveAt(index);
        }
        #endregion


        #region Events
        protected abstract void OnInitialize();
        protected abstract void ReflectFinished(Record record, long duration, int running_code);
        protected abstract void OnError(Exception ex);
        protected abstract void OnStopped();
        protected abstract void OnFinished();
        #endregion


        #region Progress Job

        // process locker is used for waiting Stop method until training task stop
        private readonly ReaderWriterLock process_locker = new ReaderWriterLock();
        public bool Canceling { get; private set; } = false;
        public bool Stopped { get; private set; } = true;
        public Task Start()
        {
            Canceling = false;
            return Task.Run(() =>
            {
                Task<Record> record_geter = null;

                try
                {
                    process_locker.AcquireWriterLock(-1);
                    Stopped = false;

                    // initialize data-provider
                    data_provider.Initialize();
                    // initialize by developer
                    OnInitialize();

                    // fetch next record
                    record_geter = data_provider.PrepareNextData(Offset, Stage);

                    // training loop
                    while (!Canceling && processes.Count > 0 && (EpochMax < 1 || Epoch < EpochMax))
                    {
                        // current record
                        record_geter.Wait();
                        var record = record_geter.Result;

                        // next record'offset and stage
                        var (next_offset, next_stage) = GetNextRound();
                        // fetch next record
                        record_geter = data_provider.PrepareNextData(next_offset, next_stage);

                        if (record != null && record.data != null && record.result != null)
                        {
                            if (Canceling) break;

                            // reporting vriables
                            var start_time = DateTime.Now.Ticks;

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
                                        break;
                                }

                            if (Canceling) break;
                            // call event
                            ReflectFinished(record, DateTime.Now.Ticks - start_time, 0);

                            if (next_offset == 0)
                                lock (processes)
                                    switch (Stage)
                                    {
                                        case TraingingStages.Training:
                                            // if next round is first round of stage
                                            // if this round is end round of stage
                                            foreach (var progress in processes)
                                                progress.FinishCurrentState(true);
                                            break;
                                        case TraingingStages.Validation:
                                            // if next round is first round of stage
                                            // if this round is end round of stage
                                            var we_have_out_of_line = false;
                                            foreach (var progress in processes)
                                                we_have_out_of_line |= progress.FinishCurrentState(false);
                                            if (!we_have_out_of_line)
                                            {
                                                // go to next epoch
                                                next_stage = TraingingStages.Training;
                                                // fetch next record
                                                record_geter = data_provider.PrepareNextData(
                                                    next_offset, next_stage);
                                            }
                                            break;
                                        case TraingingStages.Evaluation:
                                            // if next round is first round of stage
                                            // if this round is end round of stage
                                            for (int p = 0; p < processes.Count;)
                                                if (!processes[p].OutOfLine) p++;
                                                else
                                                {
                                                    lock (out_of_line)
                                                        out_of_line.Add(new BrainInfo(
                                                            processes[p].BestBrainImage,
                                                            processes[p].BestBrainAccuracy));
                                                    processes.RemoveAt(p);
                                                }
                                            // reset processs accuarcy info for next round
                                            foreach (var progress in processes)
                                                progress.FinishCurrentState(false);
                                            break;
                                    }
                        }

                        // next offset
                        if (next_offset == 0 && next_stage == TraingingStages.Training)
                            Epoch++;
                        Offset = next_offset;
                        Stage = next_stage;
                    }

                    // being sure that record geter is finished
                    _ = record_geter.Result;

                    OnFinished();
                }
                catch (Exception ex) { OnError(ex); }
                finally
                {
                    Stopped = true;
                    if (process_locker.IsWriterLockHeld)
                        process_locker.ReleaseWriterLock();
                    data_provider.Dispose();
                }
            });
        }
        public Task Evaluate()
        {
            Canceling = false;
            return Task.Run(() =>
            {
                Task<Record> record_geter = null;

                try
                {
                    process_locker.AcquireWriterLock(-1);
                    Stopped = false;

                    var EpochMax = Math.Max(this.EpochMax, 1U);
                    Stage = TraingingStages.Evaluation;

                    // initialize data-provider
                    data_provider.Initialize();
                    // initialize by developer
                    OnInitialize();

                    // fetch next record
                    record_geter = data_provider.PrepareNextData(Offset, Stage);

                    // prepare brains
                    foreach (var bri in OutOfLine)
                        bri.InitBrain();

                    // training loop
                    while (!Canceling && Epoch < EpochMax)
                    {
                        // current record
                        record_geter.Wait();
                        var record = record_geter.Result;

                        // next record'offset and stage
                        var (next_offset, next_stage) = GetNextRound();
                        // fetch next record
                        record_geter = data_provider.PrepareNextData(next_offset, next_stage);

                        if (record != null && record.data != null && record.result != null)
                        {
                            if (Canceling) break;

                            // reporting vriables
                            var start_time = DateTime.Now.Ticks;

                            lock (out_of_line)
                                Parallel.ForEach(out_of_line, (process, state, index) =>
                                {
                                    // test this neural network with evaluation data
                                    var flash = process.Brain.Test(record.data);
                                    // calculate total error
                                    process.Brain.FillTotalError(flash, record.result);
                                    // update accuracy
                                    process.ChangeSatate(flash);
                                });

                            if (Canceling) break;
                            // call event
                            ReflectFinished(record, DateTime.Now.Ticks - start_time,
                                (int)TraingingStages.Evaluation);
                        }

                        // next offset
                        if (next_offset == 0 && next_stage == TraingingStages.Training)
                            Epoch++;
                        Offset = next_offset;
                        Stage = TraingingStages.Evaluation;
                    }

                    // being sure that record geter is finished
                    _ = record_geter.Result;
                }
                catch (Exception ex) { OnError(ex); }
                finally
                {
                    Stopped = true;
                    if (process_locker.IsWriterLockHeld)
                        process_locker.ReleaseWriterLock();
                    data_provider.Dispose();
                }
            });
        }
        public void Stop()
        {
            Canceling = true;

            // wait for training task finish
            process_locker.AcquireWriterLock(-1);
            try { OnStopped(); }
            finally { process_locker.ReleaseWriterLock(); }
        }
        #endregion

        public static string GetDurationString(long duration, int level = 4)
        {
            var result = new StringBuilder();
            // 100-nanosecond
            if (level >= 6) result.Insert(0, ",").Insert(0, duration % 10000);
            // millisecond
            duration /= 10000;
            if (duration == 0) return result.Remove(result.Length - 1, 1).ToString();
            if (level >= 5) result.Insert(0, "ms,").Insert(0, duration % 1000);
            // second
            duration /= 1000;
            if (duration == 0) return result.Remove(result.Length - 1, 1).ToString();
            if (level >= 4) result.Insert(0, "s,").Insert(0, duration % 60);
            // miniute
            duration /= 60;
            if (duration == 0) return result.Remove(result.Length - 1, 1).ToString();
            if (level >= 3) result.Insert(0, "m,").Insert(0, duration % 60);
            // hour
            duration /= 60;
            if (duration == 0) return result.Remove(result.Length - 1, 1).ToString();
            if (level >= 2) result.Insert(0, "h,").Insert(0, duration % 24);
            // days
            duration /= 24;
            if (duration == 0) return result.Remove(result.Length - 1, 1).ToString();
            if (level >= 1) result.Insert(0, "d,").Insert(0, duration);
            // return
            return result.Remove(result.Length - 1, 1).ToString();
        }

    }
}