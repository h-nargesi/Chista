using System;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Photon.NeuralNetwork.Opertat.Debug
{
    public static class Setting
    {
        public const string barin_image = "barin-image";
        public const string data_provider = "data-provider";
        public const string current_offset = "current-offset";
        public const string learning_factor = "learning-factor";
        public const string certainty_factor = "certainty-factor";
        public const string dropout_factor = "dropout-factor";
        public const string learning_epoch = "learning-epoch";
        public const string learning_tries = "learning-tries";
        public const string model_layers = "model-layers";
        public const string rebuild = "rebuild";

        private static string setting_file_name = "setting.json";
        public static JObject Read(string path = null)
        {
            if (path != null) setting_file_name = path;

            using var setting_file = File.Open(setting_file_name, FileMode.OpenOrCreate);
            var buffer = new byte[setting_file.Length];
            setting_file.Read(buffer, 0, buffer.Length);
            try { return JObject.Parse(Encoding.UTF8.GetString(buffer)); }
            catch { return new JObject(); }
        }
        public static void Save(JObject json)
        {
            using StreamWriter file = File.CreateText(setting_file_name);
            using JsonTextWriter writer = new JsonTextWriter(file)
            {
                Formatting = Formatting.Indented
            };
            json.WriteTo(writer);
        }
    }
}
