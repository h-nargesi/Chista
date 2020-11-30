using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class LearningProcessInfo
    {
        public TrainingStages Stage { get; set; } = TrainingStages.Training;
        public uint Offset { get; set; }
        public uint Epoch { get; set; }
        public List<ITrainingProcess> Processes { get; set; }
        public List<IBrainInfo> OutOfLine { get; set; }
    }
}
