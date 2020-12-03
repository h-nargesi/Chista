using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class TrainingProcessHistory
    {
        public TrainingProcessHistory()
        {
            accuracy_chain = new LinkedList<double>();
        }

        private readonly LinkedList<double> accuracy_chain;
        public BrainInfo StableNetImage { get; private set; }

        public bool AddProgress(INetProcess progress)
        {
            if (accuracy_chain.Count < 1 || progress.RunningAccuracy > accuracy_chain.First.Value)
            {
                accuracy_chain.Clear();
                accuracy_chain.AddLast(progress.RunningAccuracy);
                StableNetImage = new BrainInfo(progress.RunningBrain.Image(), progress.RunningAccuracy);
                return false;
            }
            else
            {
                accuracy_chain.AddLast(progress.RunningAccuracy);

                double prv_accuracy = 0;
                int descenting_count = 0;
                foreach (var acc in accuracy_chain)
                {
                    if (prv_accuracy < acc) descenting_count = 0;
                    else descenting_count++;
                    prv_accuracy = acc;
                }

                if (descenting_count >= 4 || accuracy_chain.Count > 10)
                {
                    accuracy_chain.Clear();
                    StableNetImage = new BrainInfo(StableNetImage.Image, -1);
                    return true;
                }
                else return false;
            }
        }

        public double[] AccuracyChain()
        {
            var chain = new double[accuracy_chain.Count];
            accuracy_chain.CopyTo(chain, 0);
            return chain;
        }
        public static TrainingProcessHistory Restore(INeuralNetworkImage stable_image, double[] chain)
        {
            var history = new TrainingProcessHistory();

            if (stable_image != null)
            {
                if (chain == null || chain.Length < 1)
                    throw new ArgumentNullException(nameof(chain));

                history.StableNetImage = new BrainInfo(stable_image, chain[0]);

                foreach (var accuracy in chain)
                    history.accuracy_chain.AddLast(accuracy);
            }

            return history;
        }
    }
}
