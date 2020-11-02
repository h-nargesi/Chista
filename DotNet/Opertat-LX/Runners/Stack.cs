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

        private int company_step = 0;
        private uint last_instrument = 0;
        private List<(uint cumulative, uint company_id)> cumulative_frequency;
        private SqlCommand sqlite;
        private string print = "";
        private readonly object sqlite_lock = new object();

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

                TraingingCount = 0; ValidationCount = 0;
                cumulative_frequency = new List<(uint cumulative, uint company_id)>();
                while (reader.Read())
                {
                    cumulative_frequency.Add((TraingingCount, (uint)(int)reader[0]));
                    TraingingCount += (uint)(int)reader[1];
                    ValidationCount += (uint)(int)reader[2];
                }
                if (cumulative_frequency.Count == 0)
                    throw new Exception("The data set is empty.");

                // TODO: test validation
                // ignore validation
                ValidationCount = 0;

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
        protected override Task<Record> PrepareNextData(uint offset, bool is_training)
        {
            return Task.Run(() =>
            {
                var start_time = DateTime.Now.Ticks;
                var result = new double[RESULT_COUNT];
                var signal = new double[SIGNAL_COUNT_TOTAL];

                uint i = 0, company_id;
                (offset, company_id) = FindCompany(offset);

                lock (sqlite_lock)
                    if (sqlite != null)
                    {
                        sqlite.Parameters.Clear();
                        sqlite.Parameters.Add("@ID", SqlDbType.Int).Value = company_id;
                        sqlite.Parameters.Add("@Type", SqlDbType.Char, 1).Value = 'X';
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

                return new Record(is_training, signal, result, company_id, DateTime.Now.Ticks - start_time);
            });
        }
        private (uint offset, uint company_id) FindCompany(uint offset)
        {
            if (offset <= 0)
            {
                company_step = 0;
                return cumulative_frequency[company_step];
            }

            uint cum_left, com_left, cum_right, com_right;
            while (true)
            {
                (cum_left, com_left) = cumulative_frequency[company_step];
                if (company_step + 1 < cumulative_frequency.Count)
                    (cum_right, com_right) = cumulative_frequency[company_step + 1];
                else return (offset - cum_left, com_left);

                if (cum_left <= offset && offset < cum_right) return (offset - cum_left, com_left);
                else if (offset == cum_right)
                {
                    company_step++;
                    return (offset - cum_right, com_right);
                }
                else if (offset > com_right)
                    company_step = (cumulative_frequency.Count - (company_step + 1)) / 2;
                else if (cum_left > offset) company_step /= 2;
            }
        }
        protected override void ReflectFinished(Record record, long duration)
        {
            if (record.training && last_instrument != (uint)record.extra)
            {
                last_instrument = (uint)record.extra;
                Debugger.Console.CommitLine();
            }
            else
            {
                print = Regex.Replace(print, "[^ \t\r\n]", " ");
                Debugger.Console.WriteWord(print);
            }

            double accuracy = 0, result = 0;
            foreach (var reuslt in Brains.Values)
            {
                accuracy = Math.Max(reuslt.accuracy, accuracy);
                result += reuslt.predict.ResultSignals[0];
            }
            result /= Brains.Count;

            print = $"#{Offset / TotalCount},{Offset % TotalCount}," +
                $"{(record.training ? "training" : "validation")}:\r\n\t" +
                $"instm={record.extra}\t" +
                $"output={Print(record.result[0], 3):R}\t" +
                $"predict,avg={Print(result, 3):R}\t" +
                $"accuracy,best={Print(accuracy * 100, 4):R}\r\n\t" +
                $"data loading={GetDurationString(record.duration.Value)}\r\n\t" +
                $"prediction={GetDurationString(duration)}";

            Debugger.Console.WriteWord(print);
        }

        public override void Dispose()
        {
            base.Dispose();

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
        private static readonly int RECORDS_COUNT_BASICAL;

        static Stack()
        {
            SIGNAL_COUNT_BASICAL = SIGNAL_STEP_COUNT +
                (int)Math.Ceiling(SIGNAL_STEP_COUNT / 2.0) +
                (int)Math.Ceiling(SIGNAL_STEP_COUNT / 4.0) +
                (int)Math.Ceiling(SIGNAL_STEP_COUNT / 8.0);
            RECORDS_COUNT_BASICAL = RESULT_COUNT + SIGNAL_COUNT_BASICAL;

            SIGNAL_COUNT_LAST_YEARS = 0;
            for (int y = 1; y <= YEARS_COUNT;)
                SIGNAL_COUNT_LAST_YEARS += (int)Math.Ceiling(SIGNAL_STEP_LAST_YEARS / (double)(++y));
            SIGNAL_COUNT_TOTAL = SIGNAL_COUNT_BASICAL + SIGNAL_COUNT_LAST_YEARS;
        }

        private readonly static string sql_counting = $@"
select		InstrumentID,
            sum(iif(RecordType = 'X', 1, 0)) - {RECORDS_COUNT_BASICAL} as TrainingCount,
            sum(iif(RecordType = 'V', 1, 0)) - {RECORDS_COUNT_BASICAL} as ValidationCount
from		Trade
where		RecordType is not null
group by	InstrumentID
having      count(*) > {RECORDS_COUNT_BASICAL}
order by	TrainingCount desc";
        #endregion

    }
}