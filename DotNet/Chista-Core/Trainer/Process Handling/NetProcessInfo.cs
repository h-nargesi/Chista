using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class NetProcessInfo
    {
        public NeuralNetworkImage best_image, current_image;
        public double[] accuracy_chain;
        public int record_count;
        public double current_total_accruacy;

        public NetProcessInfo(
            NeuralNetworkImage current_image,
            int record_count,
            double current_total_accruacy,
            double[] accuracy_chain,
            NeuralNetworkImage best_image)
        {
            this.current_image = current_image;
            this.record_count = record_count;
            this.current_total_accruacy = current_total_accruacy;
            this.accuracy_chain = accuracy_chain;
            this.best_image = best_image;
        }

        public NetProcessInfo(NeuralNetworkImage image, double accruacy)
        {
            current_image = image;
            current_total_accruacy = accruacy;
            accuracy_chain = new double[0];
        }

        public ITrainingProcess TrainProcess()
        {
            return new TrainingProcess(this);
        }

        public IBrainInfo BrainInfo()
        {
            return new BrainInfo(current_image, current_total_accruacy);
        }
    }
}
