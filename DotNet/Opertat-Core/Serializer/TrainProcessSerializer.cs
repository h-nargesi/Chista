using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Photon.NeuralNetwork.Opertat.Trainer;

namespace Photon.NeuralNetwork.Opertat.Serializer
{
    public static class TrainProcessSerializer
    {
        public const string FILE_TYPE_SIGNATURE_STRING = "Opertat Training Process";
        public const byte SECTION_TYPE = 2;
        public const ushort VERSION = 4;

        public static void Serialize(string path, IReadOnlyList<IProgress> progresses)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (progresses == null)
                throw new ArgumentNullException(nameof(progresses));

            using var stream = File.Create(path);
            stream.Flush();

            // serialize file signature
            stream.Write(FILE_TYPE_SIGNATURE);

            Serialize(stream, progresses);
        }
        public static void Serialize(FileStream stream, IReadOnlyList<IProgress> progresses)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (progresses == null)
                throw new ArgumentNullException(nameof(progresses));

            byte[] buffer;

            // serialize version
            buffer = BitConverter.GetBytes(SectionType.GetSectionSign(SECTION_TYPE, VERSION)); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize progress count
            buffer = BitConverter.GetBytes(progresses.Count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize progress
            foreach (var iprg in progresses)
                if (iprg is Progress prg)
                {
                    var state = prg.Info();

                    buffer = BitConverter.GetBytes(state.record_count); // 4-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    buffer = BitConverter.GetBytes(state.current_total_accruacy); // 8-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    buffer = BitConverter.GetBytes(state.accuracy_chain.Length); // 4-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    foreach (var ac in state.accuracy_chain)
                    {
                        buffer = BitConverter.GetBytes(ac); // 8-bytes
                        stream.Write(buffer, 0, buffer.Length);
                    }

                    NeuralNetworkSerializer.Serialize(stream, state.current_image);

                    if (state.best_image == null)
                    {
                        buffer = new byte[] { 0 }; // 1-byte
                        stream.Write(buffer, 0, buffer.Length);
                    }
                    else
                    {
                        buffer = new byte[] { 1 }; // 1-byte
                        stream.Write(buffer, 0, buffer.Length);

                        NeuralNetworkSerializer.Serialize(stream, state.best_image);
                    }
                }
        }

        public static IReadOnlyList<IProgress> Restore(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var buffer = new byte[FILE_TYPE_SIGNATURE.Length];
            stream.Read(buffer, 0, buffer.Length);
            var file_type_diignature = Encoding.ASCII.GetString(buffer);

            // check file signature
            bool valid_sign;
            if (file_type_diignature != FILE_TYPE_SIGNATURE_STRING)
            {
                valid_sign = false;
                stream.Seek(0, SeekOrigin.Begin);
            }
            else valid_sign = true;

            // restore file
            return Restore(stream, valid_sign);
        }
        public static IReadOnlyList<IProgress> Restore(FileStream stream)
        {
            return Restore(stream, true);
        }
        private static IReadOnlyList<IProgress> Restore(FileStream stream, bool valid_sign)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");

            // 1: read version: 2-bytes
            var buffer = new byte[2];
            stream.Read(buffer, 0, buffer.Length);
            var (section_type, version) = SectionType.GetSectionInfo(BitConverter.ToUInt16(buffer, 0));

            if (section_type != SECTION_TYPE && version > 2)
                throw new Exception("Invalid section type");

            if (!valid_sign && version > 4)
                throw new Exception("Invalid file signature");

            switch (version)
            {
                case 2:
                case 3:
                case VERSION: return RestoreLastVersion(stream);
                default: throw new Exception("This version of progress list is not supported");
            };
        }
        private static IReadOnlyList<IProgress> RestoreLastVersion(FileStream stream)
        {
            var buffer = new byte[8];

            stream.Read(buffer, 0, 4);
            var progresses_count = BitConverter.ToInt32(buffer, 0);
            var progress = new List<Progress>(progresses_count);

            for (var i = 0; i < progresses_count; i++)
            {
                stream.Read(buffer, 0, 4);
                var record_count = BitConverter.ToInt32(buffer, 0);

                stream.Read(buffer, 0, 8);
                var current_total_accruacy = BitConverter.ToDouble(buffer, 0);

                stream.Read(buffer, 0, 4);
                var chain_count = BitConverter.ToInt32(buffer, 0);
                var accuracy_chain = new double[chain_count];

                for (var c = 0; c < chain_count; c++)
                {
                    stream.Read(buffer, 0, 8);
                    accuracy_chain[c] = BitConverter.ToDouble(buffer, 0);
                }

                var current_image = NeuralNetworkSerializer.Restore(stream);

                NeuralNetworkImage best_image;
                stream.Read(buffer, 0, 1);
                if (buffer[0] == 0) best_image = null;
                else best_image = NeuralNetworkSerializer.Restore(stream);

                progress.Add(Progress.RestoreInfo(
                    new ProgressState(
                        current_image, record_count, current_total_accruacy,
                        accuracy_chain, best_image)));
            }

            return progress;
        }


        private readonly static byte[] FILE_TYPE_SIGNATURE;
        static TrainProcessSerializer()
        {
            FILE_TYPE_SIGNATURE = Encoding.ASCII.GetBytes(FILE_TYPE_SIGNATURE_STRING);
        }
    }
}
