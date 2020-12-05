using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Chista.Implement;
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
        private void SetNewState(uint progress, TrainingStages stage)
        {
            Offset = progress;
            Stage = stage;
            if (progress == 0 && stage == TrainingStages.Training)
                Epoch++;
        }
        #endregion


        #region Chista-Net Management

        private readonly List<NetProcess> processes = new List<NetProcess>();
        private readonly List<NetProcess> out_of_lines = new List<NetProcess>();
        public IReadOnlyList<INetProcess> Processes => processes;
        public IReadOnlyList<INetProcess> OutOfLines => out_of_lines;
        public void LoadProgress(LearningProcessInfo process_info)
        {
            if (process_info == null) throw new ArgumentNullException(nameof(process_info));
            if (!Stopped) throw new Exception("The process is not stoped.");

            process_locker.AcquireWriterLock(100);
            try
            {
                processes.Clear();
                if (process_info.Processes != null)
                    foreach (NetProcess prc in process_info.Processes)
                    {
                        processes.Add(prc);
                        if (prc.RunningChistaNet == null) prc.InitialChistaNet();
                    }
                CheckBestRunningProcess();

                out_of_lines.Clear();
                if (process_info.OutOfLines != null)
                    foreach (NetProcess prc in process_info.OutOfLines)
                    {
                        out_of_lines.Add(prc);
                        if (prc.StableAccuracy < 0) prc.InitialChistaNet();
                    }
                CheckBestRunningOutOfLine();

                Epoch = process_info.Epoch;
                Stage = process_info.Stage;
                Offset = process_info.Offset;
            }
            finally { process_locker.ReleaseWriterLock(); }
        }
        public void AddRunningProgress(IChistaNet chits_net)
        {
            if (chits_net == null) throw new ArgumentNullException(nameof(chits_net));
            if (!Stopped) throw new Exception("The process is not stoped.");

            process_locker.AcquireWriterLock(100);
            try
            {
                processes.Add(new NetProcess(chits_net));
                CheckBestRunningProcess();
            }
            finally { process_locker.ReleaseWriterLock(); }
        }
        public void RemoveRunningProgress(int index)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            process_locker.AcquireWriterLock(100);
            try
            {
                processes.RemoveAt(index);
                CheckBestRunningProcess();
            }
            finally { process_locker.ReleaseWriterLock(); }
        }
        public void AddOutOfLineProgress(INeuralNetworkImage image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (!Stopped) throw new Exception("The process is not stoped.");

            process_locker.AcquireWriterLock(100);
            try
            {
                out_of_lines.Add(new NetProcess(image));
                CheckBestRunningProcess();
            }
            finally { process_locker.ReleaseWriterLock(); }
        }
        public void RemoveOutOfLine(int index)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            process_locker.AcquireWriterLock(100);
            try
            {
                out_of_lines.RemoveAt(index);
                CheckBestRunningProcess();
            }
            finally { process_locker.ReleaseWriterLock(); }
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

            process_locker.AcquireReaderLock(-1);
            try
            {
                if (BestRunningProcess != null)
                    buffer.Append("\n#best_process\t")
                        .Append(BestRunningProcess.PrintInfo());

                if (BestRunningOutOfLine != null)
                    buffer.Append("\n#best_out_of_line\t")
                        .Append(BestRunningOutOfLine.PrintInfo());
            }
            finally { process_locker.ReleaseReaderLock(); }

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
            if (!Stopped) throw new Exception("The prcoess already is running");

            Canceling = false;
            return Task.Run(() =>
            {
                Task<Record> record_geter = null;

                try
                {
                    process_locker.AcquireReaderLock(3000);
                    try
                    {
                        Stopped = false;

                        // initialize data-provider
                        data_provider.Initialize();
                        // initialize by developer
                        OnInitialize();

                        // check new out-of-line process
                        if (Stage == TrainingStages.Evaluation)
                        {
                            bool we_have_out_of_line = false;
                            foreach (var bri in out_of_lines)
                                if (bri.RunningAccuracy < 0)
                                {
                                    bri.InitialChistaNet();
                                    we_have_out_of_line = true;
                                }

                            if (!we_have_out_of_line)
                            {
                                Offset = 0;
                                Stage = TrainingStages.Training;
                            }
                        }

                        // fetch next record
                        record_geter = data_provider.PrepareNextData(Offset, Stage);

                        // training loop
                        while (!Canceling && processes.Count > 0 && (EpochMax < 1 || Epoch < EpochMax))
                        {
                            switch (Stage)
                            {
                                case TrainingStages.Training:
                                    RunTrainingStage(ref record_geter);
                                    goto case TrainingStages.Validation;
                                case TrainingStages.Validation:
                                    RunValidationStage(ref record_geter);
                                    goto case TrainingStages.Evaluation;
                                case TrainingStages.Evaluation:
                                    RunEvaluationStage(ref record_geter);
                                    break;
                            }
                        }

                    }
                    finally
                    {
                        Stopped = true;
                        process_locker.ReleaseReaderLock();

                        record_geter?.Wait();
                        data_provider.Dispose();

                        OnFinished();
                    }
                }
                catch (Exception ex) { OnError(ex); }
            });
        }
        public Task Evaluate()
        {
            if (!Stopped) throw new Exception("The prcoess already is running");

            Canceling = false;
            return Task.Run(() =>
            {
                Task<Record> record_geter = null;

                try
                {
                    process_locker.AcquireReaderLock(3000);
                    try
                    {
                        Stopped = false;

                        if (out_of_lines.Count < 1) return;
                        Stage = TrainingStages.Evaluation;
                        Offset = 0;

                        // initialize data-provider
                        data_provider.Initialize();
                        // initialize by developer
                        OnInitialize();

                        // fetch next record
                        record_geter = data_provider.PrepareNextData(Offset, Stage);

                        // prepare chista-nets
                        foreach (var bri in out_of_lines)
                        {
                            // to remove current state
                            bri.ReleaseChistaNet();
                            // initialize the chista-net again
                            bri.InitialChistaNet();
                        }

                        RunEvaluationStage(ref record_geter);
                    }
                    finally
                    {
                        Stopped = true;
                        process_locker.ReleaseReaderLock();

                        record_geter.Wait();
                        data_provider.Dispose();

                        OnFinished();
                    }
                }
                catch (Exception ex) { OnError(ex); }
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

        private void RunTrainingStage(ref Task<Record> record_geter)
        {
            // training loop
            while (!Canceling && Stage == TrainingStages.Training)
            {
                // current record
                record_geter.Wait();
                var record = record_geter.Result;

                // next record'offset and stage
                var (next_offset, next_stage) = GetNextRound();
                // fetch next record
                record_geter = data_provider.PrepareNextData(next_offset, next_stage);

                if (record == null || record.data == null || record.result == null)
                {
                    SetNewState(next_offset, next_stage);
                    continue;
                }

                // cancel before networks training
                if (Canceling) break;

                // reporting time interval
                var time_interval = DateTime.Now.Ticks;

                // CAUTION don's cancel the process in this area until to change offset
                Parallel.ForEach(processes, (process, state, index) =>
                {
                    // train this neural network
                    var flash = process.RunningChistaNet.Train(record.data, record.result);
                    // change progress state
                    process.ChangeSatate(flash);
                });

                // calculate duration time
                time_interval = DateTime.Now.Ticks - time_interval;

                if (next_offset == 0)
                    // if next round is first round of stage
                    // if this round is end round of stage
                    foreach (var progress in processes)
                        progress.FinishCurrentState(true);

                // CAUTION don's cancel the process until this area
                SetNewState(next_offset, next_stage);

                // it is safe to cancel
                if (Canceling) break;

                if (Offset % update_best_interval == 0 || next_offset == 0)
                    CheckBestRunningProcess();

                // call event
                ReflectFinished(record, time_interval, 0);
            }
        }
        private void RunValidationStage(ref Task<Record> record_geter)
        {
            // validation loop
            while (!Canceling && Stage == TrainingStages.Validation)
            {
                // current record
                record_geter.Wait();
                var record = record_geter.Result;

                // next record'offset and stage
                var (next_offset, next_stage) = GetNextRound();
                // fetch next record
                record_geter = data_provider.PrepareNextData(next_offset, next_stage);

                if (record == null || record.data == null || record.result == null)
                {
                    SetNewState(next_offset, next_stage);
                    continue;
                }

                // cancel before networks training
                if (Canceling) break;

                // reporting time interval
                var time_interval = DateTime.Now.Ticks;

                Parallel.ForEach(processes, (process, state, index) =>
                {
                    // test this neural network with validation data
                    var flash = process.RunningChistaNet.Test(record.data);
                    // calculate total error
                    process.RunningChistaNet.FillTotalError(flash, record.result);
                    // change progress state
                    process.ChangeSatate(flash);
                });

                // calculate duration time
                time_interval = DateTime.Now.Ticks - time_interval;

                if (next_offset == 0)
                {
                    var locked = process_locker.UpgradeToWriterLock(-1);
                    try
                    {
                        // if next round is first round of stage
                        // if this round is end round of stage
                        var we_have_new_out_of_line = false;
                        for (int p = 0; p < processes.Count;)
                            if (!processes[p].FinishCurrentState(false)) p++;
                            else if (processes[p].RunningChistaNet is ChistaNetLine net_line &&
                                    net_line.Index + 1 < net_line.ChistaNets.Count)
                            {
                                net_line.Index++;
                                p++;
                            }
                            else
                            {

                                // reset chista-net with stable image
                                processes[p].InitialChistaNet();
                                // move the process to out-of-line list
                                we_have_new_out_of_line = true;
                                out_of_lines.Add(processes[p]);
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
                    }
                    finally { process_locker.DowngradeFromWriterLock(ref locked); }
                }

                // CAUTION don's cancel the process until this area
                SetNewState(next_offset, next_stage);

                // it is safe to cancel
                if (Canceling) break;

                if (Offset % update_best_interval == 0 || next_offset == 0)
                    CheckBestRunningProcess();

                // call event
                ReflectFinished(record, time_interval, 0);
            }
        }
        private void RunEvaluationStage(ref Task<Record> record_geter)
        {
            // evaluation loop
            while (!Canceling && Stage == TrainingStages.Evaluation)
            {
                // current record
                record_geter.Wait();
                var record = record_geter.Result;

                // next record'offset and stage
                var (next_offset, next_stage) = GetNextRound();
                // fetch next record
                record_geter = data_provider.PrepareNextData(next_offset, next_stage);

                if (record == null || record.data == null || record.result == null)
                {
                    SetNewState(next_offset, next_stage);
                    continue;
                }

                // cancel before networks training
                if (Canceling) break;

                // reporting vriables
                var time_interval = DateTime.Now.Ticks;

                Parallel.ForEach(out_of_lines, (process, state, index) =>
                {
                    // do just new out-of-line processes
                    if (process.StableAccuracy >= 0) return;
                    // test this neural network with evaluation data
                    var flash = process.RunningChistaNet.Test(record.data);
                    // calculate total error
                    process.RunningChistaNet.FillTotalError(flash, record.result);
                    // change progress state
                    process.ChangeSatate(flash);
                });

                // calculate duration time
                time_interval = DateTime.Now.Ticks - time_interval;

                if (next_offset == 0)
                    foreach (var ool in out_of_lines)
                    {
                        ool.FinishCurrentState(false);
                        ool.ReleaseChistaNet();
                    }

                // CAUTION don's cancel the process until this area
                SetNewState(next_offset, next_stage);

                // it is safe to cancel
                if (Canceling) break;

                if (Offset % update_best_interval == 0 || next_offset == 0)
                    CheckBestRunningOutOfLine();

                // call event
                ReflectFinished(record, time_interval, 0);
            }
        }
        #endregion


        #region Reporting
        public INetProcess BestRunningProcess { get; private set; }
        public INetProcess BestRunningOutOfLine { get; private set; }

        private int update_best_interval = 1000;
        protected int UPDATE_BEST_INTERVAL
        {
            get { return update_best_interval; }
            set
            {
                if (value < 10) throw new ArgumentOutOfRangeException(
                    nameof(UPDATE_BEST_INTERVAL), "value must be greater than ten.");
                update_best_interval = value;
            }
        }

        public void CheckBestRunningProcess()
        {
            BestRunningProcess = null;
            if (processes.Count < 1) return;

            BestRunningProcess = processes[0];
            foreach (var proc in processes)
                if (BestRunningProcess.ReportingAccuracy < proc.ReportingAccuracy)
                    BestRunningProcess = proc;
        }
        public void CheckBestRunningOutOfLine()
        {
            BestRunningOutOfLine = null;
            if (out_of_lines.Count < 1) return;

            BestRunningOutOfLine = out_of_lines[0];
            foreach (var ool in out_of_lines)
                if (BestRunningOutOfLine.ReportingAccuracy < ool.ReportingAccuracy)
                    BestRunningOutOfLine = ool;
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