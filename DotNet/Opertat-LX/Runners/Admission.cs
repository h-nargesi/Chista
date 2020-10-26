using System;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    public class Admission : NeuralNetworkRunner
    {
        private string print = "";
        public const string NAME = "adm";
        private readonly Random random = new Random(DateTime.Now.Millisecond);
        public int Slowness { get; set; } = 200;
        public override string Name => NAME;
        private const int SignalRange = 100, SignalHeight = 0;

        protected override void OnInitialize()
        {
            base.OnInitialize();

            string print = null;
            var image = Brain.Image();
            for (var i = 0; i < image.layers.Length; i++)
            {
                print += Print(image.layers[i].Bias.ToArray());
                print += Print(image.layers[i].Synapse.ToArray());
                print += "\r\n";
            }

            Debugger.Console.WriteCommitLine(print);

            Epoch = uint.MaxValue;
            Count = 128;
            Offset = 0;
        }
        protected override NeuralNetworkImage BrainInitializer()
        {
            var model = GetSetting(Setting.model_layers, "2-2-1");
            var lines = model.Split('-');
            var layers = new int[lines.Length - 2];
            for (int i = 0; i < layers.Length; i++) layers[i] = int.Parse(lines[i + 1]);

            var image = new NeuralNetworkInitializer()
                .SetInputSize(int.Parse(lines[0]))
                .AddLayer(new SoftReLU(), layers)
                .AddLayer(new Sigmoind(), int.Parse(lines[^1]))
                .SetCorrection(new Errorest(),
                    new DataRange(SignalRange, SignalHeight),
                    new DataRange(SignalRange * 2, SignalRange))
                .Image();

            return image;
        }
        protected override Task<Record> PrepareNextData(uint offset)
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

        public Admission()
        {
            ReflectFinished = (flash, record, timing) =>
            {
                if (Offset % Count == 0)
                    Debugger.Console.CommitLine();
                else
                {
                    print = Regex.Replace(print, "[^ \t\r\n]", " ");
                    Debugger.Console.WriteWord(print);
                }

                print = $"#{Offset} = ";
                print += $"result:{Print(record.result, 6)}\t";
                print += $"output:{Print(flash.ResultSignals, 6)}\t";
                print += $"accuracy:{Print(Accuracy * 100, 2)}\t";
                print += $"error:{Print(Brain.Errors(flash, record.result), null)}\r\n";

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
            };
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