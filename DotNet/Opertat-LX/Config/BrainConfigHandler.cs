using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json.Linq;

namespace Photon.NeuralNetwork.Opertat.Debug.Config
{
    public class BrainConfigHandler : ConfigHandler
    {
        public BrainConfigHandler(JObject setting) : base(setting) { }

        public const string key = "brain";
        private const string images_path = "images-path";
        private const string images_count = "images-count";
        private const string learning_factor = "learning-factor";
        private const string certainty_factor = "certainty-factor";
        private const string dropout_factor = "dropout-factor";

        public string ImagesPathDefault { get; set; } = "images/";
        public string ImagesPath
        {
            get { return GetSetting(images_path, ImagesPathDefault); }
            set { SetSetting(images_path, value); }
        }

        public int ImagesCountDefault { get; set; } = 10;
        public int ImagesCount
        {
            get { return GetSetting(images_count, ImagesCountDefault); }
            set { SetSetting(images_count, value); }
        }

        public double LearningFactorDefault { get; set; } = 0.01;
        public double LearningFactor
        {
            get { return GetSetting(learning_factor, LearningFactorDefault); }
            set { SetSetting(learning_factor, value); }
        }

        public double CertaintyFactorDefault { get; set; } = 0;
        public double CertaintyFactor
        {
            get { return GetSetting(certainty_factor, CertaintyFactorDefault); }
            set { SetSetting(certainty_factor, value); }
        }

        public double DropoutFactorDefault { get; set; } = 0;
        public double DropoutFactor
        {
            get { return GetSetting(dropout_factor, DropoutFactorDefault); }
            set { SetSetting(dropout_factor, value); }
        }


        private LayersConfigHandler layer_instance;
        public LayersConfigHandler Layers
        {
            get
            {
                if (layer_instance == null)
                    layer_instance = new LayersConfigHandler(GetConfig(LayersConfigHandler.key, null));
                return layer_instance;
            }
        }

    }
}
