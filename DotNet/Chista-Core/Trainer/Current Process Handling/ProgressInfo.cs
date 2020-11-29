using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class ProgressInfo
    {
        public NeuralNetworkImage best_image, current_image;
        public double[] accuracy_chain;
        public int record_count;
        public double current_total_accruacy;
        public bool out_of_line;

        public ProgressInfo(
            NeuralNetworkImage current_image,
            int record_count,
            double current_total_accruacy,
            double[] accuracy_chain,
            NeuralNetworkImage best_image,
            bool out_of_line)
        {
            this.current_image = current_image;
            this.record_count = record_count;
            this.current_total_accruacy = current_total_accruacy;
            this.accuracy_chain = accuracy_chain;
            this.best_image = best_image;
            this.out_of_line = out_of_line;
        }

        public ITrainProcess TrainProcess()
        {
            return new TrainProcess(this);
        }
    }
}
