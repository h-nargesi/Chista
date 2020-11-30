using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class TrainingProcessHistory
    {
        public TrainingProcessHistory()
        {
            history = new LinkedList<BrainInfo>();
        }

        private readonly LinkedList<BrainInfo> history;
        public BrainInfo BestBrainInfo
        {
            get { return history.First?.Value; }
        }

        public bool AddProgress(ITrainingProcess progress)
        {
            if (history.Count < 1 || progress.Accuracy > history.First.Value.Accuracy)
            {
                history.Clear();
                history.AddLast(new BrainInfo(progress.Brain.Image(), progress.Accuracy));
                return false;
            }
            else
            {
                history.AddLast(new BrainInfo(null, progress.Accuracy));

                double prv_accuracy = 0;
                int descenting_count = 0;
                foreach (var info in history)
                {
                    if (prv_accuracy < info.Accuracy) descenting_count = 0;
                    else descenting_count++;
                    prv_accuracy = info.Accuracy;
                }

                if (descenting_count >= 4) return true;
                else if (history.Count > 10) return true;
                else return false;
            }
        }

        public double[] AccuracyChain()
        {
            var c = 0;
            var chain = new double[history.Count];
            foreach (var info in history) chain[c++] = info.Accuracy;
            return chain;
        }
        public static TrainingProcessHistory Restore(double[] chain, NeuralNetworkImage best_image)
        {
            if (chain == null)
                throw new ArgumentNullException(nameof(chain), "History.Restore: accuracy chain is null");

            var history = new TrainingProcessHistory();

            if (chain.Length > 0)
            {
                if (best_image == null)
                    throw new ArgumentNullException(nameof(best_image), "History.Restore: best_image is null");

                foreach (var ac in chain)
                {
                    history.history.AddLast(new BrainInfo(best_image, ac));
                    if (best_image != null) best_image = null;
                }
            }

            return history;
        }
    }
}
