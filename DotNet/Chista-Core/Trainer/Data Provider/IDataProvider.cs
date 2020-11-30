using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Chista.Trainer;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface IDataProvider : IDisposable
    {
        public uint TrainingCount { get; }
        public uint ValidationCount { get; }
        public uint EvaluationCount { get; }
        public void Initialize();
        public Task<Record> PrepareNextData(uint progress, TrainingStages stage);
        public string PrintInfo();
    }
}
