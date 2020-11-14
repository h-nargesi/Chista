using System;

namespace Photon.NeuralNetwork.Chista.Debug
{
    public abstract class Debugger
    {
        public abstract void WriteWord(string text);
        public abstract void CommitLine();
        public abstract void WriteCommitLine(string text);

        public static Debugger Console { get; set; }
    }
}