using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Globalization;
using System.Text.RegularExpressions;
using Photon.NeuralNetwork.Opertat.Serializer;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Debugger.Console = new CommandPrompt();

            while (true)
            {
                string process, param;
                if (args != null)
                {
                    process = args.Length > 0 ? args[0] : null;
                    param = args.Length > 1 ? args[1] : null;

                    if (param != null && param.StartsWith("-"))
                        param = param.Substring(1);
                }
                else process = param = null;

                switch (process)
                {
                    case "":
                    case null:
                        break;

                    case "quit": return;
                    case "show":
                        if (param != null)
                        {
                            if (param.StartsWith("\""))
                                param = param.Substring(1);
                            if (param.EndsWith("\""))
                                param = param.Remove(param.Length - 1);
                        }
                        string result = NeuralNetworkSerializer.Restore(param).PrintInfo();
                        result = Regex.Replace(result, "(?<!\r)\n", "\r\n");
                        Debugger.Console.WriteCommitLine(result);
                        break;

                    case Admission.NAME:
                        using (var adm = new Admission())
                            adm.Start();
                        break;
                    case Dictionary.NAME:
                        using (var dic = new Dictionary())
                            dic.Start();
                        break;
                    case Stack.NAME:
                        using (var dic = new Stack())
                            dic.Start();
                        break;

                    default:
                        Debugger.Console.WriteCommitLine("invalid process name.");
                        break;
                }

                Debugger.Console.WriteCommitLine("get process name.");
                args = Regex.Replace(Console.ReadLine().Trim(), "[ \t\r\n]{2,}", " ").Split(' ');
                Debugger.Console.CommitLine();
            }
        }

        class CommandPrompt : Debugger
        {
            public CommandPrompt()
            {
                System.Console.OutputEncoding = Encoding.UTF8;
                Commit();
            }

            private readonly object loacker = new object();
#if !LX
            private int last_line;
#else
            private char? last_char;
#endif

            private void Write(string text)
            {
#if !LX
                System.Console.SetCursorPosition(0, last_line);
                System.Console.WriteLine(text);
#else
                System.Console.Write("\r" + text);
                if (text != null && text.Length > 0)
                    last_char = text[^1];
                else last_char = null;
#endif
            }
            private void Commit()
            {
#if !LX
                last_line = System.Console.CursorTop;
                if (System.Console.CursorLeft > 0) last_line++;
#else
                if (last_char != '\n')
                    System.Console.WriteLine();
#endif
            }

            public override void WriteWord(string text)
            {
                lock (loacker) Write(text);
            }
            public override void CommitLine()
            {
                lock (loacker) Commit();
            }
            public override void WriteCommitLine(string text)
            {
                lock (loacker)
                {
                    Commit();
                    Write(text);
                    Commit();
                }
            }
        }
    }
}