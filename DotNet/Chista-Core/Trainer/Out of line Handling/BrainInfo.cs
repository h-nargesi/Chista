using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class BrainInfo : INeuralNetworkInformation
    {
        public BrainInfo(NeuralNetworkImage image, double accuracy)
        {
            this.image = image;
            Accuracy = accuracy;
        }

        public readonly NeuralNetworkImage image;
        private int record_count;
        private double total_accuracy;

        public double Accuracy { get; private set; }
        public Brain Brain { get; private set; }
        public NeuralNetworkFlash LastPrediction { get; private set; }

        public void InitBrain()
        {
            Brain = new Brain(image);
            record_count = 0;
            total_accuracy = 0;
        }
        public void ChangeSatate(NeuralNetworkFlash predict)
        {
            record_count++;
            total_accuracy += predict.Accuracy;
            Accuracy = total_accuracy / record_count;
            LastPrediction = predict;
        }

        public string PrintInfo()
        {
            return $"{image.PrintInfo()}\naccuracy: {Accuracy}";
        }
    }
}
