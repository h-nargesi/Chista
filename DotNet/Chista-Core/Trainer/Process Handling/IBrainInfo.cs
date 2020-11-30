using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public interface IBrainInfo : INetProcess
    {
        public NeuralNetworkImage Image { get; }
    }
}
