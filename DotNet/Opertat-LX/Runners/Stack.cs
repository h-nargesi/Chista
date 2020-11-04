using System;
using System.Text;
using System.Text.RegularExpressions;
using System.Data;
using System.Data.SqlClient;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using Newtonsoft.Json.Linq;
using Photon.NeuralNetwork.Opertat.Implement;
using Photon.NeuralNetwork.Opertat.Debug.Config;
using System.Runtime.InteropServices.WindowsRuntime;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    public class Stack : NeuralNetworkRunner, IDisposable
    {
        public const string NAME = "stk";
        public override string Name => NAME;

        private long offset_interval, last_printing_time = DateTime.Now.Ticks;
        private int record_count;
        private int company_step = 0, last_instrument = 0;
        private List<Step> cumulative_frequency;
        private SqlCommand sqlite;
        private string print = "";
        private readonly object sqlite_lock = new object();

        private class Step
        {
            public readonly uint start_point;
            public readonly int instrument;
            public readonly bool is_training;

            public Step(uint start_point, int instrument, bool is_training)
            {
                this.start_point = start_point;
                this.instrument = instrument;
                this.is_training = is_training;
            }
        }

        protected override void OnInitialize()
        {
            base.OnInitialize();

            lock (sqlite_lock)
            {
                sqlite = new SqlCommand
                {
                    Connection = new SqlConnection(setting.DataProvider)
                };
                sqlite.Connection.Open();

                sqlite.CommandText = sql_counting;
                using var reader = sqlite.ExecuteReader();

                Count = 0;
                cumulative_frequency = new List<Step>();
                while (reader.Read())
                {
                    var tr_count = (uint)(int)reader[1];
                    if (tr_count > 0)
                    {
                        cumulative_frequency.Add(new Step(Count, (int)reader[0], true));
                        Count += tr_count;
                    }

                    var vl_count = (uint)(int)reader[2];
                    if (vl_count > 0)
                    {
                        cumulative_frequency.Add(new Step(Count, (int)reader[0], false));
                        Count += vl_count;
                    }
                }

                if (cumulative_frequency.Count == 0)
                    throw new Exception("The data set is empty.");

                sqlite.CommandText = "GetTrade";
                sqlite.CommandType = CommandType.StoredProcedure;
            }
        }
        protected override NeuralNetworkImage[] BrainInitializer()
        {
            var layers = setting.Brain.Layers.NodesCount;
            if (layers == null || layers.Length == 0)
            {
                setting.Brain.Layers.NodesCount = new int[0];
                throw new Exception("the default layer's node count is not set.");
            }

            var images = new NeuralNetworkImage[setting.Brain.ImagesCount];
            var conduction = setting.Brain.Layers.Conduction == "soft-relu" ?
                (IConduction)new SoftReLU() : new ReLU();
            var output = setting.Brain.Layers.OutputConduction == "straight" ?
                (IConduction)new Straight() : new Sigmoind();
            var range = setting.Brain.Layers.OutputConduction == "straight" ? null : new DataRange(20, 10);

            for (int i = 0; i < images.Length; i++)
                images[i] = new NeuralNetworkInitializer()
                    .SetInputSize(SIGNAL_COUNT_TOTAL)
                    .AddLayer(conduction, layers)
                    .AddLayer(output, RESULT_COUNT)
                    .SetCorrection(new ErrorStack(RESULT_COUNT), new RegularizationL2())
                    .SetDataConvertor(new DataRange(5, 0), range)
                    .Image();

            return images;
        }
        protected override Task<Record> PrepareNextData(uint offset)
        {
            return Task.Run(() =>
            {
                var start_time = DateTime.Now.Ticks;
                var result = new double[RESULT_COUNT];
                var signal = new double[SIGNAL_COUNT_TOTAL];

                uint i = 0;
                int company_id;
                bool is_training;
                (offset, company_id, is_training) = FindCompany(offset);

                lock (sqlite_lock)
                    if (sqlite != null)
                    {
                        sqlite.Parameters.Clear();
                        sqlite.Parameters.Add("@ID", SqlDbType.Int).Value = company_id;
                        sqlite.Parameters.Add("@Type", SqlDbType.Char, 1).Value = is_training ? 'X' : 'V';
                        sqlite.Parameters.Add("@Offset", SqlDbType.Int).Value = offset;
                        using var reader = sqlite.ExecuteReader();
                        while (reader.Read())
                        {
                            if (i < RESULT_COUNT) result[i] = (double)(decimal)reader[0];
                            else if (i - RESULT_COUNT < SIGNAL_COUNT_TOTAL)
                                signal[i - RESULT_COUNT] = (double)(decimal)reader[0];
                            else break;
                            i++;
                        }
                    }

                if (i < RESULT_COUNT + SIGNAL_COUNT_TOTAL)
                    throw new Exception($"Invalid data size ({offset}).");

                return new Record(is_training, signal, result, company_id, DateTime.Now.Ticks - start_time);
            });
        }
        private (uint offset, int instrument_id, bool is_training) FindCompany(uint offset)
        {
            if (offset <= 0)
            {
                company_step = 0;
                var inst = cumulative_frequency[company_step];
                return (inst.start_point, inst.instrument, inst.is_training);
            }

            Step left, right;
            while (true)
            {
                left = cumulative_frequency[company_step];
                if (company_step + 1 < cumulative_frequency.Count)
                    right = cumulative_frequency[company_step + 1];
                else return (offset - left.start_point, left.instrument, left.is_training);

                if (left.start_point <= offset && offset < right.start_point) 
                    return (offset - left.start_point, left.instrument, left.is_training);
                else if (offset == right.start_point)
                {
                    company_step++;
                    return (offset - right.start_point, right.instrument, right.is_training);
                }
                else if (offset > right.start_point)
                    company_step = (cumulative_frequency.Count - (company_step + 1)) / 2;
                else if (left.start_point > offset) company_step /= 2;
            }
        }
        protected override void ReflectFinished(Record record, long duration)
        {
            string clearing;
            if (record.training && last_instrument != (int)record.extra)
            {
                last_instrument = (int)record.extra;
                clearing = null;
            }
            else clearing = Regex.Replace(print, "[^ \t\r\n]", " ");

            double accuracy = 0, result = 0;
            foreach (var prc in Progresses)
            {
                accuracy = Math.Max(prc.CurrentAccuracy, accuracy);
                result += prc.LastPredict.ResultSignals[0];
            }
            result /= Progresses.Count;

            offset_interval += DateTime.Now.Ticks - last_printing_time;
            record_count++;

            print = $"#{Offset / Count},{Offset % Count}:\r\n\t" +
                $"model={(record.training ? "training" : "validation")} " +
                $"{setting.Brain.ImagesCount} brain(s)\r\n\t" +
                $"instm={record.extra}\t" +
                $"accuracy,best={Print(accuracy * 100, 4):R}\r\n\t" +
                $"output={Print(record.result[0], 3):R}\t" +
                $"predict,avg={Print(result, 3):R}\r\n\t" +
                $"data loading={GetDurationString(record.duration.Value)}\t" +
                $"prediction={GetDurationString(duration)}\r\n" +
                $"*\tleft-time={GetDurationString(offset_interval / record_count * (Count - Offset))}";

            last_printing_time = DateTime.Now.Ticks;

            if (clearing == null) Debugger.Console.CommitLine();
            else Debugger.Console.WriteWord(clearing);
            Debugger.Console.WriteWord(print);
        }
        protected override void OnStopped()
        {
            base.OnStopped();

            if (sqlite != null)
                lock (sqlite_lock)
                {
                    var connection = sqlite.Connection;
                    sqlite.Dispose();
                    connection?.Dispose();
                    sqlite = null;
                }
        }


        #region SQL Queries
        private const int YEARS_COUNT = 3;
        private const int RESULT_COUNT = 20;
        private const int SIGNAL_STEP_COUNT = 40;
        private const int SIGNAL_STEP_LAST_YEARS = RESULT_COUNT + SIGNAL_STEP_COUNT;
        private static readonly int SIGNAL_COUNT_BASICAL;
        private static readonly int SIGNAL_COUNT_LAST_YEARS;
        private static readonly int SIGNAL_COUNT_TOTAL;

        static Stack()
        {
            SIGNAL_COUNT_BASICAL = SIGNAL_STEP_COUNT +
                (int)Math.Ceiling(SIGNAL_STEP_COUNT / 2.0) +
                (int)Math.Ceiling(SIGNAL_STEP_COUNT / 4.0) +
                (int)Math.Ceiling(SIGNAL_STEP_COUNT / 8.0);

            SIGNAL_COUNT_LAST_YEARS = 0;
            for (int y = 1; y <= YEARS_COUNT;)
                SIGNAL_COUNT_LAST_YEARS += (int)Math.Ceiling(SIGNAL_STEP_LAST_YEARS / (double)(++y));
            SIGNAL_COUNT_TOTAL = SIGNAL_COUNT_BASICAL + SIGNAL_COUNT_LAST_YEARS;
        }

        private readonly static string sql_counting = $@"
select		InstrumentID,
            sum(iif(RecordType = 'X', 1, 0)) as TrainingCount,
            sum(iif(RecordType = 'V', 1, 0)) as ValidationCount
from		Trade
where		RecordType is not null
group by	InstrumentID
having      sum(iif(RecordType = 'X', 1, 0)) > 0
        and sum(iif(RecordType = 'V', 1, 0)) > 0
order by    InstrumentID";
        #endregion

    }
}