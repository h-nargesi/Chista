using System;
using System.Collections.Generic;
using System.Text;
using Photon.NeuralNetwork.Chista;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Debug.Tools
{
    static class FunctionDecoder
    {
        public static IConduction Conduction(string key)
        {
            return key switch
            {
                "sigmoind" => (IConduction)new Sigmoind(),
                "soft-relu" => new SoftReLU(),
                "relu" => new ReLU(),
                "soft-max" => new SoftMax(),
                _ => throw new Exception("invalid conduction function")
            };
        }

        public static IErrorFunction ErrorFunction(string key)
        {
            return key switch
            {
                "errorest" => (IErrorFunction)new Errorest(),
                "cross-entropy" => new CrossEntropy(),
                "classification" => new Classification(),
                "tagging" => new Tagging(0.8, 0.4),
                _ => throw new Exception("invalid error function")
            };
        }
    }
}
