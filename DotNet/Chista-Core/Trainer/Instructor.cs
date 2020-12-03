using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Chista.Trainer.Delegates;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public abstract class Instructor : INeuralNetworkInformation
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

                Stage = TrainingStages.Training;
                Offset = 0;
                Epoch = 0;

                data_provider = value;
            }
        }
        public TrainingStages Stage { get; set; } = TrainingStages.Training;
        public uint Offset { get; set; }
        public uint Epoch { get; set; }
        public uint EpochMax { get; set; }
        private (uint progress, TrainingStages stage) GetNextRound()
        {
            var progress = Offset + 1;
            switch (Stage)
            {
                case TrainingStages.Training:
                    if (progress >= data_provider.TrainingCount)
                        return (0, TrainingStages.Validation);
                    else return (progress, Stage);

                case TrainingStages.Validation:
                    if (progress >= data_provider.ValidationCount)
                        return (0, TrainingStages.Evaluation);
                    else return (progress, Stage);

                case TrainingStages.Evaluation:
                    if (progress >= data_provider.EvaluationCount)
                        return (0, TrainingStages.Training);
                    else return (progress, Stage);

                default: throw new Exception($"Invalid stage! ({Stage})");
            }
        }
        #endregion


        #region Brain Management

        private readonly List<NetProcess> processes = new List<NetProcess>();
        private readonly List<NetProcess> out_of_line = new List<NetProcess>();
        public IReadOnlyList<INetProcess> Processes => processes;
        public IReadOnlyList<INetProcess> OutOfLine => out_of_line;
        public void LoadProgress(LearningProcessInfo process_info)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            if (process_info == null)
                throw new ArgumentNullException(nameof(process_info));

            lock (processes)
            {
                processes.Clear();
                if (process_info.Processes != null)
                    foreach (NetProcess prc in process_info.Processes)
                    {
                        processes.Add(prc);
                        if (prc.RunningBrain == null) prc.InitialBrain();
                    }
            }

            lock (out_of_line)
            {
                out_of_line.Clear();
                if (process_info.OutOfLine != null)
                    foreach (NetProcess prc in process_info.OutOfLine)
                    {
                        out_of_line.Add(prc);
                        if (prc.StableAccuracy < 0) prc.InitialBrain();
                    }
            }

            Epoch = process_info.Epoch;
            Stage = process_info.Stage;
            Offset = process_info.Offset;
        }
        public void AddProgress(Brain brain)
        {
            if (brain == null) throw new ArgumentNullException(nameof(brain));
            lock (processes) processes.Add(new NetProcess(brain));
        }
        public void RemoveProgress(int index)
        {
            lock (processes) processes.RemoveAt(index);
        }
        public void AddBrainInfo(Brain brain)
        {
            if (brain == null) throw new ArgumentNullException(nameof(brain));
            lock (out_of_line) out_of_line.Add(new NetProcess(brain));
        }
        public void RemoveBrainInfo(int index)
        {
            lock (out_of_line) out_of_line.RemoveAt(index);
        }
        public string PrintInfo()
        {
            var buffer = new StringBuilder("[instructor]");

            buffer
                .Append("\n").Append(data_provider.PrintInfo())
                .Append("\n").Append("epoch: #").Append(Epoch);
            if (Epoch > 0) buffer.Append(" to ").Append(EpochMax);
            buffer
                .Append(", stage: ").Append(Stage.ToString().ToLower())
                .Append(", offset: ").Append(Offset);

            lock (processes)
                if (processes.Count > 0)
                {
                    int best_index = -1, current_index = -1; double best_accuracy = -1;
                    foreach (var prc in processes)
                    {
                        current_index++;
                        if (best_accuracy >= prc.StableAccuracy) continue;

                        best_accuracy = prc.StableAccuracy;
                        best_index = current_index;
                    }

                    buffer.Append("\n#best process\n")
                        .Append(processes[best_index].PrintInfo());
                }

            lock (out_of_line)
                if (out_of_line.Count > 0)
                {
                    int best_index = -1, current_index = -1; double best_accuracy = -1;
                    foreach (var prc in out_of_line)
                    {
                        current_index++;
                        if (best_accuracy >= prc.RunningAccuracy) continue;

                        best_accuracy = prc.RunningAccuracy;
                        best_index = current_index;
                    }

                    buffer.Append("\n#best out_of_line\n")
                        .Append(out_of_line[best_index].PrintInfo());
                }

            return buffer.ToString();
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

                    // check new out-of-line process
                    if (Stage == TrainingStages.Evaluation)
                    {
                        var offset = Offset;
                        Offset = 0;
                        Stage = TrainingStages.Training;
                        lock (out_of_line)
                            foreach (var bri in out_of_line)
                                if (bri.RunningAccuracy < 0)
                                {
                                    Offset = offset;
                                    Stage = TrainingStages.Evaluation;
                                    break;
                                }
                    }

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

                            switch (Stage)
                            {
                                case TrainingStages.Training:
                                    lock (processes)
                                        Parallel.ForEach(processes, (process, state, index) =>
                                        {
                                            // train this neural network
                                            var flash = process.RunningBrain.Train(record.data, record.result);
                                            // change progress state
                                            process.ChangeSatate(flash);
                                        });
                                    break;
                                case TrainingStages.Validation:
                                    lock (processes)
                                        Parallel.ForEach(processes, (process, state, index) =>
                                        {
                                            // test this neural network with validation data
                                            var flash = process.RunningBrain.Test(record.data);
                                            // calculate total error
                                            process.RunningBrain.FillTotalError(flash, record.result);
                                            // change progress state
                                            process.ChangeSatate(flash);
                                        });
                                    break;
                                case TrainingStages.Evaluation:
                                    lock (out_of_line)
                                        Parallel.ForEach(out_of_line, (process, state, index) =>
                                        {
                                            // do just new out-of-line processes
                                            if (process.RunningAccuracy >= 0) return;
                                            // for sure
                                            if (process.RunningBrain == null) process.InitialBrain();
                                            // test this neural network with evaluation data
                                            var flash = process.RunningBrain.Test(record.data);
                                            // calculate total error
                                            process.RunningBrain.FillTotalError(flash, record.result);
                                            // change progress state
                                            process.ChangeSatate(flash);
                                        });
                                    break;
                            }

                            if (Canceling) break;
                            // call event
                            ReflectFinished(record, DateTime.Now.Ticks - start_time, 0);

                            if (next_offset == 0)
                                switch (Stage)
                                {
                                    case TrainingStages.Training:
                                        // if next round is first round of stage
                                        // if this round is end round of stage
                                        lock (processes)
                                            foreach (var progress in processes)
                                                progress.FinishCurrentState(true);
                                        break;
                                    case TrainingStages.Validation:
                                        // if next round is first round of stage
                                        // if this round is end round of stage
                                        var we_have_new_out_of_line = false;
                                        lock (processes)
                                            for (int p = 0; p < processes.Count;)
                                                if (!processes[p].FinishCurrentState(false)) p++;
                                                else
                                                {
                                                    we_have_new_out_of_line = true;
                                                    lock (out_of_line) out_of_line.Add(processes[p]);
                                                    processes.RemoveAt(p);
                                                }
                                        // skip next evaluation round if we do not have out-of-line process
                                        if (!we_have_new_out_of_line)
                                        {
                                            // go to next epoch
                                            next_stage = TrainingStages.Training;
                                            // fetch next record
                                            record_geter = data_provider.PrepareNextData(
                                                next_offset, next_stage);
                                        }
                                        break;
                                    case TrainingStages.Evaluation:
                                        lock (out_of_line)
                                            foreach (var ool in out_of_line)
                                                ool.ReleaseBrain();
                                        break;
                                }
                        }

                        // next offset
                        if (next_offset == 0 && next_stage == TrainingStages.Training)
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
                    Stage = TrainingStages.Evaluation;

                    // initialize data-provider
                    data_provider.Initialize();
                    // initialize by developer
                    OnInitialize();

                    // fetch next record
                    record_geter = data_provider.PrepareNextData(Offset, Stage);

                    // prepare brains
                    lock (out_of_line)
                        foreach (var bri in out_of_line) bri.InitialBrain();

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
                                    var flash = process.RunningBrain.Test(record.data);
                                    // calculate total error
                                    process.RunningBrain.FillTotalError(flash, record.result);
                                    // update accuracy
                                    process.ChangeSatate(flash);
                                });

                            if (Canceling) break;
                            // call event
                            ReflectFinished(record, DateTime.Now.Ticks - start_time,
                                (int)TrainingStages.Evaluation);
                        }

                        // next offset
                        if (next_offset == 0 && next_stage == TrainingStages.Training)
                            Epoch++;
                        Offset = next_offset;
                        Stage = TrainingStages.Evaluation;
                    }

                    // prepare brains
                    lock (out_of_line)
                        foreach (var bri in out_of_line) bri.ReleaseBrain();

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
        public void Stop()
        {
            Canceling = true;

            // wait for training task finish
            process_locker.AcquireWriterLock(10000);
            try { OnStopped(); }
            finally { process_locker.ReleaseWriterLock(); }
        }
        #endregion


        public static string GetDurationString(long duration, int level = 4)
        {
            var result = GetDurationStringEnded(duration, level);
            if (result.Length < 1)
                return level switch
                {
                    1 => "less than 1d",
                    2 => "less than 1h",
                    3 => "less than 1m",
                    4 => "less than 1s",
                    5 => "less than 1ms",
                    _ => "immediate",
                };
            else return result.Remove(result.Length - 1, 1).ToString();
        }
        private static StringBuilder GetDurationStringEnded(long duration, int level)
        {
            var result = new StringBuilder();
            // 100-nanosecond
            if (level >= 6) result.Insert(0, ",").Insert(0, duration % 10000);
            // millisecond
            duration /= 10000;
            if (duration == 0) return result;
            if (level >= 5) result.Insert(0, "ms,").Insert(0, duration % 1000);
            // second
            duration /= 1000;
            if (duration == 0) return result;
            if (level >= 4) result.Insert(0, "s,").Insert(0, duration % 60);
            // miniute
            duration /= 60;
            if (duration == 0) return result;
            if (level >= 3) result.Insert(0, "m,").Insert(0, duration % 60);
            // hour
            duration /= 60;
            if (duration == 0) return result;
            if (level >= 2) result.Insert(0, "h,").Insert(0, duration % 24);
            // days
            duration /= 24;
            if (duration == 0) return result;
            if (level >= 1) result.Insert(0, "d,").Insert(0, duration);
            // return
            return result;
        }

    }
}