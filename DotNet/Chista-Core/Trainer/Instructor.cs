using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Chista.Trainer.Delegates;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class Instructor
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
        public void LoadProgress(ProcessInfo process_info)
        {
            if (!Stopped) throw new Exception("The process is not stoped.");

            if (process_info == null)
                throw new ArgumentNullException(nameof(process_info));

            processes.Clear();
            foreach (var iprg in process_info.Processes)
                if (iprg is TrainProcess prg)
                    processes.Add(prg);
                else throw new Exception("Invalid progress type.");

            out_of_line.Clear();
            if (out_of_line != null)
                foreach (var ibrain in process_info.OutOfLine)
                    if (ibrain is BrainInfo brn)
                        out_of_line.Add(brn);
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


        #region Events
        public event OnInitializeHandler OnInitialize;
        public event ReflectFinishedHandler ReflectFinished;
        public event OnErrorHandler OnError;
        public event OnStoppedHandler OnStopped;
        #endregion


        #region Progress Job

        // process locker is used for waiting Stop method until training task stop
        private readonly ReaderWriterLock process_locker = new ReaderWriterLock();
        public bool Canceling { get; private set; } = false;
        public bool Stopped => process_locker.IsWriterLockHeld;
        public Task Start()
        {
            return Task.Run(() =>
            {
                Canceling = false;
                process_locker.AcquireWriterLock(4096);
                Task<Record> record_geter = null;

                try
                {
                    // initialize data-provider
                    data_provider.Initialize();
                    // initialize by developer
                    OnInitialize?.Invoke(this);

                    // fetch next record
                    record_geter = data_provider.PrepareNextData(Offset, Stage);

                    // training loop
                    while (!Canceling && processes.Count > 0)
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
                            ReflectFinished?.Invoke(this, record, DateTime.Now.Ticks - start_time);

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
                                            foreach (var progress in processes)
                                                progress.FinishCurrentState(false);
                                            // if next round is first round of stage
                                            // if this round is end round of stage
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
                        }

                        // next offset
                        if (next_offset == 0 && next_stage == TraingingStages.Training)
                            Epoch++;
                        Offset = next_offset;
                        Stage = next_stage;
                    }
                }
                catch (Exception ex) { OnError?.Invoke(this, ex); }
                finally { process_locker.ReleaseWriterLock(); }
            });
        }
        public void Stop()
        {
            Canceling = true;
            if (OnStopped != null)
            {
                // wait for training task finish
                process_locker.AcquireWriterLock(4096);
                try
                {
                    data_provider.Dispose();
                    OnStopped(this);
                }
                finally { process_locker.ReleaseWriterLock(); }
            }
        }
        #endregion


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