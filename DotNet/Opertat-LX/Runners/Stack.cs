using System;
using System.Text;
using System.Data;
using System.Data.SqlClient;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using Newtonsoft.Json.Linq;
using System.Text.RegularExpressions;

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

        protected override void OnInitialize()
        {
            base.OnInitialize();

            sqlite = new SqlCommand
            {
                Connection =
                    new SqlConnection(GetSetting(Setting.data_provider, ""))
            };
            sqlite.Connection.Open();

            sqlite.CommandText = sql_counting;
            using var reader = sqlite.ExecuteReader();

            Count = 0;
            cumulative_frequency = new List<(uint cumulative, uint company_id)>();
            while (reader.Read())
            {
                cumulative_frequency.Add((Count, (uint)(int)reader[0]));
                Count += (uint)(int)reader[1];
            }
            if (cumulative_frequency.Count == 0)
                throw new Exception("The data set is empty.");

            sqlite.CommandText = "GetTrade";
            sqlite.CommandType = CommandType.StoredProcedure;
        }
        protected override NeuralNetworkImage BrainInitializer()
        {
            var relu = new ReLU();
            var init = new NeuralNetworkInitializer()
                .SetInputSize(SIGNAL_COUNT_TOTAL);

            // hard code layer size
            init.AddLayer(relu, 100, 100, 100);

            init.AddLayer(new Sigmoind(), RESULT_COUNT)
                .SetCorrection(new ErrorStack(RESULT_COUNT))
                .SetDataConvertor(new DataRange(5, 0), new DataRange(10, 5));

            return init.Image();
        }
        protected override async Task<Record> PrepareNextData(uint offset)
        {
            var start_time = DateTime.Now.Ticks;
            var result = new double[RESULT_COUNT];
            var signal = new double[SIGNAL_COUNT_TOTAL];

            uint i = 0, company_id;
            (offset, company_id) = FindCompany(offset);

            sqlite.Parameters.Clear();
            sqlite.Parameters.Add("@ID", SqlDbType.Int).Value = company_id;
            sqlite.Parameters.Add("@Type", SqlDbType.Char, 1).Value = 'X';
            sqlite.Parameters.Add("@Offset", SqlDbType.Int).Value = offset;
            using var reader = await sqlite.ExecuteReaderAsync();
            while (reader.Read())
            {
                if (i < RESULT_COUNT) result[i] = (double)(decimal)reader[1];
                else if (i - RESULT_COUNT < SIGNAL_COUNT_TOTAL)
                    signal[i - RESULT_COUNT] = (double)(decimal)reader[1];
                else break;
                i++;
            }

            return new Record(signal, result, company_id, DateTime.Now.Ticks - start_time);
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

        public Stack()
        {
            ReflectFinished = (flash, record, duration) =>
            {
                if (last_instrument != (uint)record.extra)
                {
                    last_instrument = (uint)record.extra;
                    Debugger.Console.CommitLine();
                }
                else
                {
                    print = Regex.Replace(print, "[^ \t\r\n]", " ");
                    Debugger.Console.WriteWord(print);
                }

                print = $"#{Offset / Count},{Offset % Count}:\r\n\t" +
                    $"instrument={record.extra}\t" +
                    $"output={Print(record.result[0], 3):R}\t" +
                    $"predict={Print(flash.ResultSignals[0], 3):R}\t" +
                    $"accuracy={Print(Accuracy * 100, 4):R}\r\n\t" +
                    $"data loading={GetDurationString(record.duration.Value)}\r\n\t" +
                    $"prediction={GetDurationString(duration)}";

                Debugger.Console.WriteWord(print);
            };
        }

        public override void Dispose()
        {
            base.Dispose();

            if (sqlite != null)
            {
                var connection = sqlite.Connection;
                sqlite.Dispose();
                connection?.Dispose();
                sqlite = null;
            }
        }


        #region SQL Queries
        private const int RESULT_COUNT = 20;
        private const int SIGNAL_STEP_COUNT = 40;
        private const int SIGNAL_LAST_YEARS = SIGNAL_STEP_COUNT + RESULT_COUNT;
        private const int SIGNAL_COUNT_BASICAL = RESULT_COUNT + SIGNAL_STEP_COUNT * 4;
        private const int SIGNAL_COUNT_TOTAL = SIGNAL_COUNT_BASICAL + SIGNAL_LAST_YEARS * YEARS_COUNT;
        private const int YEARS_COUNT = 3;

        private readonly static string sql_counting = $@"
select		InstrumentID, sum(iif(RecordType = 'X', 1, 0)) - {SIGNAL_COUNT_BASICAL} as Amount
from		Trade
where		RecordType is not null
group by	InstrumentID
having      count(*) > {SIGNAL_COUNT_BASICAL}
order by	Amount desc";
        #endregion
    }
}