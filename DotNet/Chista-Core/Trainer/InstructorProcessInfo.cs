using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class InstructorProcessInfo
    {
        public TraingingStages Stage { get; set; } = TraingingStages.Training;
        public uint Offset { get; set; }
        public uint Epoch { get; set; }
        public List<ITrainProcess> Processes { get; set; }
        public List<BrainInfo> OutOfLine { get; set; }
    }
}
