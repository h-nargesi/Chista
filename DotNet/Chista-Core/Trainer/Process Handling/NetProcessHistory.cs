using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    class NetProcessHistory
    {
        public NetProcessHistory()
        {
            accuracy_chain = new LinkedList<double>();
        }

        private readonly LinkedList<double> accuracy_chain;
        public NetProcessImage StableNetImage { get; private set; }

        public bool AddProgress(INetProcess progress)
        {
            if (StableNetImage != null && progress.RunningAccuracy < accuracy_chain.First?.Value)
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
                    StableNetImage = new NetProcessImage(StableNetImage.Image, -1);
                    return true;
                }
                else return false;
            }
            else
            {
                bool is_stable;
                if (StableNetImage == null || accuracy_chain.Count > 0)
                {
                    accuracy_chain.Clear();
                    accuracy_chain.AddLast(progress.RunningAccuracy);
                    is_stable = false;
                }
                else is_stable = true;

                StableNetImage = new NetProcessImage(
                    progress.RunningChistaNet.Image(), progress.RunningAccuracy);

                return is_stable;
            }
        }

        public double[] AccuracyChain()
        {
            var chain = new double[accuracy_chain.Count];
            accuracy_chain.CopyTo(chain, 0);
            return chain;
        }
        public static NetProcessHistory Restore(INeuralNetworkImage stable_image, double[] chain)
        {
            var history = new NetProcessHistory();

            if (stable_image != null)
            {
                if (chain == null || chain.Length < 1)
                    throw new ArgumentNullException(nameof(chain));

                history.StableNetImage = new NetProcessImage(stable_image, chain[0]);

                foreach (var accuracy in chain)
                    history.accuracy_chain.AddLast(accuracy);
            }

            return history;
        }
        public static NetProcessHistory Restore(INeuralNetworkImage stable_image)
        {
            var history = new NetProcessHistory();

            if (stable_image != null)
                history.StableNetImage = new NetProcessImage(stable_image, -1);

            return history;
        }
    }
}
