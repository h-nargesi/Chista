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
        protected readonly ConfigHandler setting;
        public NeuralNetworkRunner()
        {
            setting = new ConfigHandler(Setting.Read($"setting-{Name}.json"));
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

            Offset = setting.GetSetting<uint>(Setting.current_offset, 0);
            Epoch = setting.GetSetting<uint>(Setting.learning_epoch, 1024);
            Tries = setting.GetSetting<uint>(Setting.learning_tries, 1);

            bool? rebuild = setting.GetSetting<bool>(Setting.rebuild);
            string barin_image = setting.GetSetting(Setting.barin_image, $"{Name}.nni");
            if (File.Exists(barin_image) && rebuild != true)
            {
                Debugger.Console.WriteCommitLine("loading brain ... ");
                Brain = new Brain(NeuralNetworkSerializer.Restore(barin_image));
            }
            else
            {
                Offset = 0;
                Debugger.Console.WriteCommitLine("new brain ... ");
                Brain = new Brain(BrainInitializer());
            }

            if (rebuild == true) setting.SetSetting(Setting.rebuild, false);

            Brain.LearningFactor = setting.GetSetting(Setting.learning_factor, 0.1);
            Brain.CertaintyFactor = setting.GetSetting(Setting.certainty_factor, 0.001);
            Brain.DropoutFactor = setting.GetSetting(Setting.dropout_factor, 0.2);
        }
        protected abstract NeuralNetworkImage BrainInitializer();
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

            setting.SetSetting(Setting.current_offset, Offset);

            Debugger.Console.WriteCommitLine("storing brain's image ... ");
            NeuralNetworkSerializer.Serialize(
                Brain.Image(),
                setting.GetSetting(Setting.barin_image, $"{Name}.nni")
            );

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