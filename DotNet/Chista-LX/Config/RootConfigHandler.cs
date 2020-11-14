using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Photon.NeuralNetwork.Chista.Debug.Config
{
    public class RootConfigHandler : ConfigHandler
    {
        public RootConfigHandler(string path = null) : base(Read(path)) { }


        private static string setting_file_name = "setting.json";
        private static JObject Read(string path = null)
        {
            if (path != null) setting_file_name = path;

            using var setting_file = File.Open(setting_file_name, FileMode.OpenOrCreate);
            var buffer = new byte[setting_file.Length];
            setting_file.Read(buffer, 0, buffer.Length);
            try { return JObject.Parse(Encoding.UTF8.GetString(buffer)); }
            catch { return new JObject(); }
        }
        public void Save()
        {
            using StreamWriter file = File.CreateText(setting_file_name);
            using JsonTextWriter writer = new JsonTextWriter(file)
            {
                Formatting = Formatting.Indented
            };
            setting.WriteTo(writer);
        }


        private const string data_provider = "data-provider";
        public string DataProvider
        {
            get { return GetSetting(data_provider, ""); }
            set { SetSetting(data_provider, value); }
        }

        private BrainConfigHandler brain_instance;
        public BrainConfigHandler Brain
        {
            get
            {
                if (brain_instance == null)
                    brain_instance = new BrainConfigHandler(GetConfig(BrainConfigHandler.key, null));
                return brain_instance;
            }
        }

        private ProcessConfigHandler process;
        public ProcessConfigHandler Process
        {
            get
            {
                if (process == null)
                    process = new ProcessConfigHandler(GetConfig(ProcessConfigHandler.key, null));
                return process;
            }
        }
    }
}
