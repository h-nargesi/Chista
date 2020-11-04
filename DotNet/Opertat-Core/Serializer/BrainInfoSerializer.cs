using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Photon.NeuralNetwork.Opertat.Trainer;

namespace Photon.NeuralNetwork.Opertat.Serializer
{
    public static class BrainInfoSerializer
    {
        public const ushort VERSION = 1;

        public static void Serialize(string path, IReadOnlyList<BrainInfo> out_of_line)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (out_of_line == null)
                throw new ArgumentNullException(nameof(out_of_line));

            using var stream = File.Create(path);
            stream.Flush();
            Serialize(stream, out_of_line);
        }
        public static void Serialize(FileStream stream, IReadOnlyList<BrainInfo> out_of_line)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (out_of_line == null)
                throw new ArgumentNullException(nameof(out_of_line));

            byte[] buffer;

            // serialize version
            buffer = BitConverter.GetBytes(VERSION); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize brain count
            buffer = BitConverter.GetBytes(out_of_line.Count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize progress
            foreach (var brn in out_of_line)
            {
                if (brn.image == null)
                    throw new Exception("The brain's image is null");

                buffer = BitConverter.GetBytes(brn.accuracy); // 8-bytes
                stream.Write(buffer, 0, buffer.Length);

                NeuralNetworkSerializer.Serialize(stream, brn.image);
            }
        }

        public static IReadOnlyList<BrainInfo> Restore(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);
            return Restore(stream);
        }
        public static IReadOnlyList<BrainInfo> Restore(FileStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");

            // 1: read version: 2-bytes
            var buffer = new byte[2];
            stream.Read(buffer, 0, buffer.Length);
            var version = BitConverter.ToUInt16(buffer, 0);

            return version switch
            {
                VERSION => RestoreLastVersion(stream),
                _ => throw new Exception("This version of progress list is not supported"),
            };
        }
        private static IReadOnlyList<BrainInfo> RestoreLastVersion(FileStream stream)
        {
            var buffer = new byte[8];

            stream.Read(buffer, 0, 4);
            var brains_count = BitConverter.ToInt32(buffer, 0);
            var brains = new List<BrainInfo>(brains_count);

            for (var i = 0; i < brains_count; i++)
            {
                stream.Read(buffer, 0, 8);
                var accruacy = BitConverter.ToDouble(buffer, 0);

                var image = NeuralNetworkSerializer.Restore(stream);

                brains.Add(new BrainInfo(image, accruacy));
            }

            return brains;
        }
    }
}
