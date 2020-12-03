using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public class NetProcessInfo
    {
        public INeuralNetworkImage stable_image, running_image;
        public double[] accuracy_chain_history;
        public int running_record_count;
        public double running_total_accruacy;

        public NetProcessInfo(
            INeuralNetworkImage running_image,
            int running_record_count,
            double running_total_accruacy,
            INeuralNetworkImage stable_image,
            double[] accuracy_chain_history)
        {
            this.running_image = running_image;
            this.running_record_count = running_record_count;
            this.running_total_accruacy = running_total_accruacy;
            this.stable_image = stable_image;
            this.accuracy_chain_history = accuracy_chain_history;
        }

        public INetProcess TrainProcess()
        {
            return new NetProcess(this);
        }
    }
}
