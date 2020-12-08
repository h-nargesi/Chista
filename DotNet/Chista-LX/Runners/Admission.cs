using System;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;
using Photon.NeuralNetwork.Chista.Trainer;

namespace Photon.NeuralNetwork.Chista.Debug
{
    class Admission2 : NetProcessRunner
    {
        public int Slowness { get; set; } = 200;
        public override string Name => NAME;

        public const string NAME = "adm";
        private const int SignalRange = 100, SignalHeight = 0;

        private string print = "";
        private readonly Random random = new Random(DateTime.Now.Millisecond);

        protected override void OnInitialize()
        {
            setting.Brain.ImagesPathDefault = "";
            base.OnInitialize();

            string print = null;
            var image = Processes[0].Brain.Image();
            for (var i = 0; i < image.layers.Length; i++)
            {
                print += Print(image.layers[i].Bias.ToArray());
                print += Print(image.layers[i].Synapse.ToArray());
                print += "\r\n";
            }

            Debugger.Console.WriteCommitLine(print);

            Epoch = 0;
            Offset = 0;
            TrainingCount = 128;
            ValidationCount = 0;
            EvaluationCount = 0;
        }
        protected override NeuralNetworkImage[] BrainInitializer()
        {
            var conduction = setting.Brain.Layers.Conduction;
            var layers = setting.Brain.Layers.NodesCount;
            if (layers == null || layers.Length == 0)
            {
                setting.Brain.Layers.NodesCount = new int[0];
                throw new Exception("the default layer's node count is not set.");
            }

            var image = new NeuralNetworkInitializer()
                .SetInputSize(2)
                .AddLayer(conduction == "soft-relu" ? (IConduction)new SoftReLU() : new ReLU(), layers)
                .AddLayer(new Sigmoind(), 1)
                .SetCorrection(new Errorest())
                .SetDataConvertor(
                    new DataRange(SignalRange, SignalHeight),
                    new DataRange(SignalRange * 2, SignalRange))
                .Image();

            return new NeuralNetworkImage[] { image };
        }
        protected override Task<Record> PrepareNextData(uint offset, TraingingStages stage)
        {
            double[] data = new double[] {
                random.NextDouble() * SignalRange * 2 - SignalRange,
                random.NextDouble() * SignalRange * 2 - SignalRange
            };

            double[] result = new double[] {
                (
                    (data[0] * 8 + data[1] * 2 - 2) * 3 +
                    (data[0] * 3 + data[1] * 7 - 1) * 1 + 7
                ) / 40
            };

            return Task.FromResult(new Record(data, result));
        }
        protected override void ReflectFinished(Record record, long duration, int running_code)
        {
            if (Offset == 0)
                Debugger.Console.CommitLine();
            else
            {
                print = Regex.Replace(print, "[^ \t\r\n]", " ");
                Debugger.Console.WriteWord(print);
            }

            var accuracy = Processes[0].CurrentAccuracy;
            var predict = Processes[0].LastPredict;

            print = $"#{Offset} = ";
            print += $"result:{Print(record.result, 6)}\t";
            print += $"output:{Print(predict.ResultSignals, 6)}\t";
            print += $"accuracy:{Print(accuracy * 100, 2)}\t";
            print += $"error:{Print(Processes[0].Brain.Errors(predict, record.result), null)}\r\n";

            /*var image = Brain.Image();
            for (var i = 0; i < image.layers.Length; i++)
            {
                print += Print(image.layers[i].Bias.ToArray());
                print += Print(image.layers[i].Synapse.ToArray());
                print += "\r\n";
            }*/

            Debugger.Console.WriteWord(print[0..^2]);

            if (Slowness > 1)
                Thread.Sleep(Slowness - 1);
        }
        protected override void UserControl()
        {
            Debugger.Console.WriteCommitLine("press escape key to exit.");
            do
            {
                var key = Console.ReadKey(true).Key;
                switch (key)
                {
                    case ConsoleKey.Escape: return;
                    case ConsoleKey.Add:
                        Slowness = Math.Max(Slowness / 2, 1);
                        break;
                    case ConsoleKey.Subtract:
                        Slowness = Math.Min(Slowness * 2, 5000);
                        break;
                    case ConsoleKey.Spacebar:
                        Slowness = 1;
                        break;
                    case ConsoleKey.Backspace:
                        Slowness = 5000;
                        break;
                }
            }
            while (!Stopped);
        }

        private double Sigmoind(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
        private double[] Sigmoind(double[] input)
        {
            Vector<double> vector = Vector<double>.Build.DenseOfArray(input);
            return (1 / (1 + (vector * -1).PointwiseExp())).ToArray();
        }

        protected override void OnFinished()
        {
            throw new NotImplementedException();
        }
    }
}