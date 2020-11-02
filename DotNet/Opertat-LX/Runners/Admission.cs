using System;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;
using Photon.NeuralNetwork.Opertat.Debug.Config;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    public class Admission : NeuralNetworkRunner
    {
        public int Slowness { get; set; } = 200;
        public override string Name => NAME;

        public const string NAME = "adm";
        private const int SignalRange = 100, SignalHeight = 0;

        private Brain brain;
        private string print = "";
        private readonly Random random = new Random(DateTime.Now.Millisecond);

        protected override void OnInitialize()
        {
            setting.Brain.ImagesPathDefault = "";
            base.OnInitialize();

            brain = null;
            foreach (var b in Brains.Keys) brain = b;

            string print = null;
            var image = brain.Image();
            for (var i = 0; i < image.layers.Length; i++)
            {
                print += Print(image.layers[i].Bias.ToArray());
                print += Print(image.layers[i].Synapse.ToArray());
                print += "\r\n";
            }

            Debugger.Console.WriteCommitLine(print);

            Epoch = uint.MaxValue;
            TraingingCount = 128;
            ValidationCount = 0;
            Offset = 96;
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
                .SetCorrection(new Errorest(), new RegularizationL1())
                .SetDataConvertor(
                    new DataRange(SignalRange, SignalHeight),
                    new DataRange(SignalRange * 2, SignalRange))
                .Image();

            return new NeuralNetworkImage[] { image };
        }
        protected override Task<Record> PrepareNextData(uint offset, bool training)
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

            return Task.FromResult(new Record(training, data, result));
        }
        protected override void ReflectFinished(Record record, long duration)
        {
            if (Offset % TotalCount == 0)
                Debugger.Console.CommitLine();
            else
            {
                print = Regex.Replace(print, "[^ \t\r\n]", " ");
                Debugger.Console.WriteWord(print);
            }

            var (accuracy, predict) = Brains[brain];

            print = $"#{Offset} = ";
            print += $"result:{Print(record.result, 6)}\t";
            print += $"output:{Print(predict.ResultSignals, 6)}\t";
            print += $"accuracy:{Print(accuracy * 100, 2)}\t";
            print += $"error:{Print(brain.Errors(predict, record.result), null)}\r\n";

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
            while (!Disposed);
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
    }
}