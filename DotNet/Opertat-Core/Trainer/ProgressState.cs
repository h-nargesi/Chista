using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Opertat.Trainer
{
    class ProgressState
    {
        public readonly NeuralNetworkImage best_image, current_image;
        public readonly double[] accuracy_chain;
        public readonly int record_count;
        public readonly double current_total_accruacy;

        public ProgressState(
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
    }
}
