using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Globalization;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Debugger.Console = new CommandPrompt();
            Debugger.Console.CommitLine();

            string process = args != null && args.Length > 0 ? args[0] : null;

            while (true)
            {
                switch (process?.ToLower()?.Substring(1))
                {
                    case "": break;
                    case "quit": return;

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
                        Debugger.Console.CommitLine();
                        Debugger.Console.WriteCommitLine("invalid process name.");
                        break;
                }

                process = "-" + Console.ReadLine();
                Debugger.Console.CommitLine();
            }
        }

        class CommandPrompt : Debugger
        {
            public CommandPrompt()
            {
                System.Console.OutputEncoding = Encoding.UTF8;
            }

#if !LX
            private int last_line;
#endif

            public override void WriteWord(string text)
            {
#if !LX
                System.Console.SetCursorPosition(0, last_line);
                System.Console.WriteLine(text);
#else
                System.Console.Write("\r" + text);
#endif
            }
            public override void CommitLine()
            {
#if !LX
                last_line = System.Console.CursorTop;
#else
                System.Console.WriteLine();
#endif
            }
            public override void WriteCommitLine(string text)
            {
                WriteWord(text);
                CommitLine();
            }
        }
    }
}