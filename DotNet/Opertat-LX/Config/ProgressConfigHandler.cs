using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;

namespace Photon.NeuralNetwork.Opertat.Debug.Config
{
    public class ProgressConfigHandler : ConfigHandler
    {
        public ProgressConfigHandler(JObject setting) : base(setting) { }

        public const string key = "progress";
        public const string current_offset = "current-offset";
        public const string learning_epoch = "learning-epoch";
        public const string learning_tries = "learning-tries";
        public const string rebuild = "rebuild";

        public uint CurrentOffsetDefault { get; set; } = 0;
        public uint CurrentOffset
        {
            get { return GetSetting(current_offset, CurrentOffsetDefault); }
            set { SetSetting(current_offset, value); }
        }

        public uint LearningEpochDefault { get; set; } = 1024;
        public uint LearningEpoch
        {
            get { return GetSetting(learning_epoch, LearningEpochDefault); }
            set { SetSetting(learning_epoch, value); }
        }

        public uint LearningTriesDefault { get; set; } = 1;
        public uint LearningTries
        {
            get { return GetSetting(learning_tries, LearningTriesDefault); }
            set { SetSetting(learning_tries, value); }
        }

        public bool Rebuild
        {
            get { return GetSetting<bool>(rebuild) ?? false; }
            set { SetSetting(rebuild, value); }
        }

    }
}
