﻿using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class TrainProcess : ITrainProcess
    {
        private readonly History history;
        private int record_count;
        private double total_accuracy;

        public TrainProcess(Brain brain)
        {
            Brain = brain ??
                throw new ArgumentNullException(nameof(brain), "Instructor.Progress: brain is null");
            history = new History();
        }
        public TrainProcess(ProgressInfo state)
        {
            if (state == null)
                throw new ArgumentNullException(nameof(state), "The state is null.");

            Brain = new Brain(state.current_image);
            history = History.Restore(state.accuracy_chain, state.best_image);
            record_count = state.record_count;
            total_accuracy = state.current_total_accruacy;
            OutOfLine = state.out_of_line;
        }

        public Brain Brain { get; }
        public double CurrentAccuracy { get; private set; }
        public NeuralNetworkFlash LastPredict { get; private set; }
        public bool OutOfLine { get; private set; }

        public void ChangeSatate(NeuralNetworkFlash predict)
        {
            record_count++;
            total_accuracy += predict.Accuracy;
            CurrentAccuracy = total_accuracy / record_count;
            LastPredict = predict;
        }
        public bool FinishCurrentState(bool is_training)
        {
            if (!is_training)
                OutOfLine = history.AddProgress(this);

            record_count = 0;
            total_accuracy = 0;
            CurrentAccuracy = 0;

            return !is_training && OutOfLine;
        }

        public NeuralNetworkImage BestBrainImage
        {
            get { return history.BestBrainInfo?.image; }
        }
        public double BestBrainAccuracy
        {
            get { return history.BestBrainInfo?.Accuracy ?? 0; }
        }

        public ProgressInfo ProgressInfo()
        {
            return new ProgressInfo(Brain.Image(), record_count, total_accuracy,
                history.AccuracyChain(), history.BestBrainInfo?.image, OutOfLine);
        }

        public override string ToString()
        {
            return $"accuracy: {CurrentAccuracy}, (flash:{LastPredict})";
        }
        public string PrintInfo()
        {
            return $"{Brain.PrintInfo()}\ncurrent accuracy: {CurrentAccuracy}";
        }
    }
}
