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
        public List<INetProcess> Processes { get; set; }
        public List<INetProcess> OutOfLine { get; set; }
    }
}
