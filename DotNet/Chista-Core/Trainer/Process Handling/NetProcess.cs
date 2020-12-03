using Photon.NeuralNetwork.Chista.Implement;
using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class NetProcess : INetProcess
    {
        private readonly NetProcessHistory history;
        private int record_count;
        private double total_accruacy;

        public NetProcess(IChistaNet chista_net)
        {
            RunningChistaNet = chista_net ?? throw new ArgumentNullException(nameof(chista_net));
            history = new NetProcessHistory();
        }
        public NetProcess(INeuralNetworkImage image)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image));
            history = NetProcessHistory.Restore(image);
        }
        public NetProcess(NetProcessInfo state)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));

            history = NetProcessHistory.Restore(state.stable_image, state.accuracy_chain_history);

            if (state.running_image is NeuralNetworkImage image)
                RunningChistaNet = new ChistaNet(image);
            else if (state.running_image is NeuralNetworkLineImage line_image)
                RunningChistaNet = new ChistaNetLine(line_image);

            else if (state.stable_image == null)
                throw new ArgumentException(nameof(state),
                    "The state must have stable-image or running-image");

            else return;

            record_count = state.running_record_count;
            total_accruacy = state.running_total_accruacy;
            if (record_count > 0)
                RunningAccuracy = total_accruacy / record_count;
        }

        public IChistaNet RunningChistaNet { get; private set; }
        public double RunningAccuracy { get; private set; }
        public NeuralNetworkFlash LastPrediction { get; private set; }

        public void InitialChistaNet()
        {
            if (StableImage is NeuralNetworkImage image)
                RunningChistaNet = new ChistaNet(image);
            else if (StableImage is NeuralNetworkLineImage line_image)
                RunningChistaNet = new ChistaNetLine(line_image);

            else throw new Exception("The stable image does not exist.");
        }
        public void ChangeSatate(NeuralNetworkFlash predict)
        {
            record_count++;
            total_accruacy += predict.Accuracy;
            RunningAccuracy = total_accruacy / record_count;
            LastPrediction = predict;
        }
        public bool FinishCurrentState(bool is_training)
        {
            record_count = 0;
            total_accruacy = 0;

            if (is_training) return false;
            else return history.AddProgress(this);
        }
        public void ReleaseChistaNet()
        {
            RunningChistaNet = null;
            record_count = 0;
            total_accruacy = 0;
        }

        public INeuralNetworkImage StableImage
        {
            get { return history.StableNetImage?.Image; }
        }
        public double StableAccuracy
        {
            get { return history.StableNetImage?.Accuracy ?? 0; }
        }

        public NetProcessInfo ProcessInfo()
        {
            return new NetProcessInfo(RunningChistaNet?.Image(), record_count, total_accruacy,
                history.StableNetImage?.Image, history.AccuracyChain());
        }

        public override string ToString()
        {
            return $"accuracy: {RunningAccuracy}, (flash:{LastPrediction})";
        }
        public string PrintInfo()
        {
            return $"{RunningChistaNet.PrintInfo()}\ncurrent accuracy: {RunningAccuracy}";
        }
    }
}
