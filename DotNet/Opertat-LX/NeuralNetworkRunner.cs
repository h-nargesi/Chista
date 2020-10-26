using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    public abstract class NeuralNetworkRunner : Instructor, IDisposable
    {
        protected readonly JObject setting;
        public NeuralNetworkRunner()
        {
            setting = Setting.Read($"setting-{Name}.json");
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

            Offset = GetSetting<uint>(Setting.current_offset, 0);
            Epoch = GetSetting<uint>(Setting.learning_epoch, 1024);
            Tries = GetSetting<uint>(Setting.learning_tries, 1);

            bool? rebuild = GetSetting<bool>(Setting.rebuild);
            string barin_image = GetSetting(Setting.barin_image, $"{Name}.nni");
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

            if (rebuild == true) SetSetting(Setting.rebuild, false);

            Brain.LearningFactor = GetSetting(Setting.learning_factor, 0.1);
            Brain.CertaintyFactor = GetSetting(Setting.certainty_factor, 0);
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

            SetSetting(Setting.current_offset, Offset);

            Debugger.Console.WriteCommitLine("storing brain's image ... ");
            NeuralNetworkSerializer.Serialize(
                Brain.Image(),
                GetSetting(Setting.barin_image, $"{Name}.nni")
            );

            Setting.Save(setting);
            Disposed = true;

            Debugger.Console.WriteCommitLine("finished");
        }

        public abstract string Name { get; }
        public bool Disposed { get; protected set; }

        protected T? GetSetting<T>(string name) where T : struct
        {
            if (!setting.ContainsKey(name)) return null;
            else return setting.Value<T>(name);
        }
        protected T GetSetting<T>(string name, T default_value)
        {
            T value;
            if (!setting.ContainsKey(name))
            {
                value = default_value;
                setting.Add(name, JToken.FromObject(value));
            }
            else value = setting.Value<T>(name);

            return value;
        }
        protected void SetSetting(string name, object value)
        {
            if (!setting.ContainsKey(name))
                setting.Add(name, JToken.FromObject(value));
            else setting[name].Replace(JToken.FromObject(value));
        }
        protected T[] GetSettingArray<T>(string name, T[] default_value)
        {
            T[] value;
            if (!setting.ContainsKey(name))
            {
                value = default_value;
                setting.Add(name, JArray.FromObject(value));
            }
            else
            {
                var array = setting.Value<JArray>(name);
                value = array.Select(jv => jv.Value<T>()).ToArray();
            }

            return value;
        }

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