using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class BrainInfo : IBrainInfo
    {
        public BrainInfo(NeuralNetworkImage image, double accuracy)
        {
            Image = image;
            Accuracy = accuracy;
        }

        private int record_count;
        private double total_accuracy;

        public NeuralNetworkImage Image { get; }
        public double Accuracy { get; private set; }
        public Brain Brain { get; private set; }
        public NeuralNetworkFlash LastPrediction { get; private set; }

        public void InitBrain()
        {
            Brain = new Brain(Image);
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
        public void ReleaseBrain()
        {
            Brain = null;
        }

        public NetProcessInfo ProcessInfo()
        {
            return new NetProcessInfo(Image, Accuracy);
        }

        public string PrintInfo()
        {
            return $"{Image.PrintInfo()}\naccuracy: {Accuracy}";
        }
    }
}
