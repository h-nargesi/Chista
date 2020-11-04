using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Opertat.Trainer
{
    class Progress : IProgress
    {
        private readonly History history;
        private int record_count;
        private double total_accuracy;

        public Progress(Brain brain)
        {
            Brain = brain ??
                throw new ArgumentNullException(nameof(brain), "Instructor.Progress: brain is null");
            history = new History();
        }
        private Progress(Brain brain, History history)
        {
            Brain = brain ??
                throw new ArgumentNullException(nameof(brain), "Instructor.Progress: brain is null");
            this.history = history;
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
            get { return history.BestBrainInfo?.image; }
        }
        public double BestBrainAccuracy
        {
            get { return history.BestBrainInfo.accuracy; }
        }

        public ProgressState Info()
        {
            return new ProgressState(Brain.Image(), record_count, total_accuracy,
                history.AccuracyChain(), history.BestBrainInfo?.image);
        }
        public static Progress RestoreInfo(ProgressState state)
        {
            if (state == null)
                throw new ArgumentNullException(nameof(state), "The state is null.");

            var progress = new Progress(
                new Brain(state.current_image),
                History.Restore(state.accuracy_chain, state.best_image));
            progress.record_count = state.record_count;
            progress.total_accuracy = state.current_total_accruacy;

            return progress;
        }
    }
}
