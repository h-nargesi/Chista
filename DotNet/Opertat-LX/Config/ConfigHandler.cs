﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Newtonsoft.Json.Linq;

namespace Photon.NeuralNetwork.Opertat.Debug.Config
{
    public class ConfigHandler
    {
        protected readonly JObject setting;
        public ConfigHandler(JObject setting)
        {
            this.setting = setting ??
                throw new ArgumentNullException(nameof(setting));
        }

        public ConfigHandler this[string name, object default_value]
        {
            get
            {
                if (!setting.ContainsKey(name))
                {
                    JObject obj;
                    if (default_value == null) obj = new JObject();
                    else obj = new JObject(default_value);
                    setting.Add(name, obj);
                    return new ConfigHandler(obj);
                }
                else return new ConfigHandler(setting.Value<JObject>(name));
            }
        }
        public ConfigHandler this[string name]
        {
            get
            {
                if (!setting.ContainsKey(name)) return null;
                else return new ConfigHandler(setting.Value<JObject>(name));
            }
        }

        public T? GetSetting<T>(string name) where T : struct
        {
            if (!setting.ContainsKey(name)) return null;
            else return setting.Value<T>(name);
        }
        public T GetSetting<T>(string name, T default_value)
        {
            T value;
            if (!setting.ContainsKey(name))
            {
                value = default_value;
                setting.Add(name, JToken.FromObject(value));
            }
            else value = setting.Value<T>(name);

            return value;
        }
        public T[] GetSettingArray<T>(string name, params T[] default_value)
        {
            T[] value;
            if (!setting.ContainsKey(name))
            {
                value = default_value;
                setting.Add(name, JArray.FromObject(value));
            }
            else
            {
                var array = setting.Value<JArray>(name);
                value = array.Select(jv => jv.Value<T>()).ToArray();
            }

            return value;
        }
        public T[] GetSettingArray<T>(string name)
        {
            if (!setting.ContainsKey(name)) return null;
            else
            {
                var array = setting.Value<JArray>(name);

                T[] value;
                value = array.Select(jv => jv.Value<T>()).ToArray();
                return value;
            }
        }
        public void SetSetting(string name, object value)
        {
            if (!setting.ContainsKey(name))
                setting.Add(name, JToken.FromObject(value));
            else setting[name].Replace(JToken.FromObject(value));
        }

        public void Save()
        {
            Setting.Save(setting);
        }
    }
}