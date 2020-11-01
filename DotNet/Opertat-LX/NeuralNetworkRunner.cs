using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Opertat.Debug.Config;

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

            Offset = setting.Progress.CurrentOffset;
            Epoch = setting.Progress.LearningEpoch;
            Tries = setting.Progress.LearningTries;


            NeuralNetworkImage[] images;
            if (!setting.Progress.Rebuild)
            {
                string path = setting.Brain.ImagesPath;
                if (string.IsNullOrWhiteSpace(path)) path = Directory.GetCurrentDirectory();
                else path = path.Trim();

                string[] file_names;
                if (!Directory.Exists(path)) file_names = null;
                else file_names = Directory.GetFiles(path, $"{Name}-*.nni");

                if (file_names == null || file_names.Length == 0) images = null;
                else
                {
                    Debugger.Console.WriteCommitLine("loading brain ... ");
                    images = new NeuralNetworkImage[file_names.Length];

                    Parallel.ForEach(file_names, (file, state, index) =>
                        images[index] = NeuralNetworkSerializer.Restore(file));
                }
            }
            else
            {
                images = null;
                setting.Progress.Rebuild = false;
            }

            if (images == null)
            {
                Offset = 0;
                Debugger.Console.WriteCommitLine("new brain ... ");
                images = BrainInitializer();
            }

            foreach (var image in images)
                BrainAdd(new Brain(image)
                {
                    LearningFactor = setting.Brain.LearningFactor,
                    CertaintyFactor = setting.Brain.CertaintyFactor,
                    DropoutFactor = setting.Brain.DropoutFactor,
                });
        }
        protected abstract NeuralNetworkImage[] BrainInitializer();
        protected override void OnError(Exception ex)
        {
            Debugger.Console.CommitLine();
            Debugger.Console.WriteCommitLine(ex.Message);
            Debugger.Console.WriteCommitLine(ex.StackTrace);
        }
        protected virtual void UserControl()
        {
            Debugger.Console.WriteCommitLine("press escape key to exit.");
            while (Console.ReadKey(true).Key != ConsoleKey.Escape) ;
        }
        public override void Dispose()
        {
            base.Dispose();

            Debugger.Console.CommitLine();
            Debugger.Console.WriteCommitLine("finishing ... ");

            setting.Progress.CurrentOffset = Offset;

            Debugger.Console.WriteCommitLine("storing brain's image ... ");
            string image_file_name = $"{Name}-?.nni";

            Directory.CreateDirectory(setting.Brain.ImagesPath);
            Parallel.ForEach(Brains.Keys, (brain, state, index) =>
                NeuralNetworkSerializer.Serialize(
                   brain.Image(),
                   setting.Brain.ImagesPath + image_file_name.Replace("?", (index + 1).ToString())));

            setting.Save();
            Disposed = true;

            Debugger.Console.WriteCommitLine("finished");
        }

        public abstract string Name { get; }
        public bool Disposed { get; protected set; }

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