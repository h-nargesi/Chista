using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Opertat.Debug.Config;
using Photon.NeuralNetwork.Opertat.Trainer;
using Photon.NeuralNetwork.Opertat.Serializer;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    public abstract class NeuralNetworkRunner : Instructor, IDisposable
    {
        protected readonly RootConfigHandler setting;
        public NeuralNetworkRunner()
        {
            setting = new RootConfigHandler($"setting-{Name}.json");
        }

        public new void Start()
        {
            base.Start();
            UserControl();
        }

        protected override void OnInitialize()
        {
            Debugger.Console.WriteCommitLine("initializing ... ");

            CultureInfo ci = new CultureInfo("en-GB");
            ci.NumberFormat.NumberDecimalSeparator = ".";
            Thread.CurrentThread.CurrentCulture = ci;
            Thread.CurrentThread.CurrentUICulture = ci;

            if (string.IsNullOrWhiteSpace(setting.Brain.ImagesPath))
                setting.Brain.ImagesPath = $"{Name}.nnp";
            if (!setting.Brain.Rebuild && File.Exists(setting.Brain.ImagesPath))
            {
                Debugger.Console.WriteCommitLine("loading brain ... ");
                TrainProcessSerializer.Restore(setting.Brain.ImagesPath, this);

                // reset stage
                var stage = setting.Process.Stage;
                if (stage != null)
                {
                    Stage = stage.Value;
                    setting.Process.Stage = null;
                }

                // reset offsets
                var offset = setting.Process.Offset;
                if (offset != null)
                {
                    Offset = offset.Value;
                    setting.Process.Offset = null;
                }
            }
            else
            {
                Debugger.Console.WriteCommitLine("new brain ... ");

                if (setting.Brain.Rebuild)
                {
                    setting.Brain.Rebuild = false;
                    Offset = 0;
                }

                var images = BrainInitializer();
                foreach (var image in images)
                    AddProgress(new Brain(image)
                    {
                        LearningFactor = setting.Brain.LearningFactor,
                        CertaintyFactor = setting.Brain.CertaintyFactor,
                        DropoutFactor = setting.Brain.DropoutFactor,
                    });
            }
        }
        protected abstract NeuralNetworkImage[] BrainInitializer();
        protected override void OnError(Exception ex)
        {
            Debugger.Console.WriteCommitLine(ex.Message);
            Debugger.Console.WriteCommitLine(ex.StackTrace);
        }
        protected virtual void UserControl()
        {
            Debugger.Console.WriteCommitLine("press escape key to exit.");
            while (Console.ReadKey(true).Key != ConsoleKey.Escape) ;
        }
        protected override void OnStopped()
        {
            Debugger.Console.WriteCommitLine("storing brain's image ... ");
            if (string.IsNullOrWhiteSpace(setting.Brain.ImagesPath))
                setting.Brain.ImagesPath = $"{Name}.nnp";
            TrainProcessSerializer.Serialize(setting.Brain.ImagesPath, this);

            setting.Save();

            Debugger.Console.WriteCommitLine("finished");
        }

        public abstract string Name { get; }

        protected static string Print(double[,] matrix, int? digit = 6)
        {
            if (matrix is null)
                throw new ArgumentNullException(nameof(matrix));

            var buffer = new StringBuilder();
            buffer.Append("[");
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                buffer.Append("(");
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    if (j > 0) buffer.Append(",");
                    buffer.Append(Print(matrix[i, j], digit));
                }
                buffer.Append(")");
            }
            buffer.Append("]");
            return buffer.ToString();
        }
        protected static string Print(double[] vector, int? digit = 6)
        {
            if (vector is null)
                throw new ArgumentNullException(nameof(vector));

            var buffer = new StringBuilder();
            buffer.Append("(");
            for (int j = 0; j < vector.Length; j++)
            {
                if (j > 0) buffer.Append(",");
                buffer.Append(Print(vector[j], digit));
            }
            buffer.Append(")");

            return buffer.ToString();
        }
        protected static string Print(double val, int? digit)
        {
            if (digit.HasValue)
                val = Math.Round(val, digit.Value);
            return (val >= 0 ? "+" : "") + val.ToString("R");
        }

    }
}