using System;
using System.Text;
using System.Text.RegularExpressions;
using System.Data.SQLite;
using System.Threading.Tasks;
using Photon.NeuralNetwork.Chista.Trainer;
using Photon.NeuralNetwork.Chista.Debug.Config;

namespace Photon.NeuralNetwork.Chista.Debug
{
    class Dictionary : NetProcessRunner
    {
        private const int LAYER_COUBNT = 5;
        public const string NAME = "dic";
        private string print = "";
        public override string Name => NAME;

        public Dictionary() : base(new WordsDataProvider())
        {
            ((WordsDataProvider)DataProvider).Setting = setting;
        }

        protected override NeuralNetworkImage[] BrainInitializer()
        {
            var relu = new SoftReLU();
            var init = new NeuralNetworkInitializer()
                .SetInputSize(20);

            for (int i = 1; i < LAYER_COUBNT - 1; i++)
                init.AddLayer(relu, LAYER_COUBNT + 40 - (i + 1));

            init.AddLayer(new Sigmoind(), 40)
                .SetCorrection(new Errorest())
                .SetDataConvertor(new DataRange(128, -128), new DataRange(255, 0));

            return new NeuralNetworkImage[] { (NeuralNetworkImage)init.Image() };
        }
        protected double[] CheckResultValidation(NeuralNetworkFlash flash, Record record)
        {
            var action_str = Encoding.UTF8.GetString(Action(flash.ResultSignals)).Trim();
            var f = action_str.IndexOf('\0');
            if (f >= 0) action_str = action_str.Substring(0, f);

            Clear();

            var extra = record.extra as string[];
            if (extra[1] == action_str) return null;

            IgnoreUnnecessary(record.result, flash.ResultSignals);

            print = $"{Offset}: ({extra[0]},{Finglish(Encoding.UTF8.GetString(Action(record.result)).Trim())}) => ({Print(flash.TotalError, null)}){Finglish(Trim(action_str))}";
            Debugger.Console.WriteWord(print);

            return record.result;
        }
        protected override void ReflectFinished(Record record, long duration, int running_code)
        {
            if (Offset == 0)
            {
                print = "";
                if (record == null) return;

                var extra = record.extra as string[];
                Debugger.Console.CommitLine();
                Debugger.Console.WriteWord($"{Offset}: {extra[0]}");
            }
        }
        protected override void OnFinished() { }

        public void IgnoreUnnecessary(double[] answer, double[] result)
        {
            bool first_zero = false;
            for (int i = 0; i < answer.Length; i++)
                if (first_zero) answer[i] = result[i];
                else if (answer[i] == 0) first_zero = true;
        }
        public byte[] Action(double[] signals)
        {
            if (signals == null) return null;
            var result = new byte[signals.Length];
            var i = 0;
            foreach (var s in signals)
                result[i++] = (byte)(s < 256 ? s : 255);
            return result;
        }
        public string Trim(string text)
        {
            var result = new StringBuilder(text);

            for (int i = 0; i < 32; i++)
                result.Replace("" + (char)i, "");
            for (int i = 0x7F; i <= 0xA0; i++)
                result.Replace("" + (char)i, "");

            return result.ToString().Trim();
        }
        public string Finglish(string text)
        {
            var result = new StringBuilder(text);

            result.Replace("ا", "a");
            result.Replace("آ", "a");
            result.Replace("ب", "b");
            result.Replace("پ", "p");
            result.Replace("ت", "t");
            result.Replace("ث", "th");
            result.Replace("ج", "j");
            result.Replace("چ", "ch");
            result.Replace("ح", "h");
            result.Replace("خ", "x");
            result.Replace("د", "d");
            result.Replace("ذ", "z");
            result.Replace("ر", "r");
            result.Replace("ز", "z");
            result.Replace("ژ", "zh");
            result.Replace("س", "s");
            result.Replace("ش", "sh");
            result.Replace("ص", "s");
            result.Replace("ض", "z");
            result.Replace("ط", "t");
            result.Replace("ظ", "z");
            result.Replace("ع", "a");
            result.Replace("غ", "gh");
            result.Replace("ک", "k");
            result.Replace("گ", "g");
            result.Replace("ل", "l");
            result.Replace("م", "m");
            result.Replace("ن", "n");
            result.Replace("و", "v");
            result.Replace("ه", "h");
            result.Replace("ی", "i");

            return result.ToString();
        }

        private void Clear()
        {
            print = Regex.Replace(print, "[^ \t\r\n]", " ");
            Debugger.Console.WriteWord(print);
        }

        private class WordsDataProvider : IDataProvider
        {
            private SQLiteCommand sqlite;
            private const string sql_selection =
                "select * from En_Pr order by lower(English) limit 1 offset ";

            public uint TrainingCount { get; private set; }
            public uint ValidationCount { get; private set; }
            public uint EvaluationCount { get; private set; }
            public RootConfigHandler Setting { get; set; }

            public void Initialize()
            {

                sqlite = new SQLiteCommand(new SQLiteConnection(Setting.DataProvider));
                sqlite.Connection.Open();

                sqlite.CommandText = "select count(*) from En_Pr";
                using var reader = sqlite.ExecuteReader();
                if (reader.Read()) TrainingCount = (uint)(long)reader[0];
                else throw new Exception("The unkown count.");

                ValidationCount = 0;
                EvaluationCount = 0;
            }
            public void Dispose()
            {
                if (sqlite != null)
                {
                    var connection = sqlite.Connection;
                    sqlite.Dispose();
                    connection?.Dispose();
                    sqlite = null;
                }
            }

            public async Task<Record> PrepareNextData(uint progress, TrainingStages stage)
            {
                var start = DateTime.Now.Ticks;
                byte[] result = null, data = null;
                string result_str = null, data_str = null;

                sqlite.CommandText = sql_selection + progress;
                using (var reader = await sqlite.ExecuteReaderAsync())
                {
                    if (reader.Read())
                    {
                        data_str = reader[0] as string;
                        if (string.IsNullOrEmpty(data_str)) return null;
                        else
                        {
                            data = Encoding.ASCII.GetBytes(data_str);

                            if (data.Length > 20) return null;

                            result_str = reader[1] as string;
                            if (!string.IsNullOrEmpty(result_str))
                            {
                                result = Encoding.UTF8.GetBytes(result_str);
                                if (result.Length > 40) return null;
                            }
                        }
                    }
                    else return null;
                }

                return new Record(
                    Sense(data, 20), Sense(result, 40), DateTime.Now.Ticks - start,
                    new string[] { data_str, result_str });
            }
            public double[] Sense(byte[] data, int size)
            {
                if (data == null) return null;
                var result = new double[size];
                var i = 0;
                foreach (var d in data) result[i++] = d;
                return result;
            }

            public string PrintInfo()
            {
                return ToString();
            }
            public override string ToString()
            {
                return "WordsDataProvider";
            }

        }
    }
}