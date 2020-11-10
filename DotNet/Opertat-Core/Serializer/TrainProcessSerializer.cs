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
        public const ushort VERSION = 5;

        public static void Serialize(string path, Instructor instructor)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (instructor == null)
                throw new ArgumentNullException(nameof(instructor));

            using var stream = File.Create(path);
            stream.Flush();

            // serialize file signature
            stream.Write(FILE_TYPE_SIGNATURE);

            Serialize(stream, instructor);
        }
        public static void Serialize(FileStream stream, Instructor instructor)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (instructor == null)
                throw new ArgumentNullException(nameof(instructor));

            byte[] buffer;

            // serialize version
            buffer = BitConverter.GetBytes(SectionType.GetSectionSign(SECTION_TYPE, VERSION)); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize process info
            buffer = BitConverter.GetBytes(instructor.Epoch); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);
            // serialize process info
            buffer = BitConverter.GetBytes(instructor.Offset); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);
            // serialize process info
            buffer = new byte[1] { (byte)instructor.Stage }; // 1-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize process count
            buffer = BitConverter.GetBytes(instructor.Processes.Count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize process
            foreach (var iprc in instructor.Processes)
                if (iprc is TrainProcess prc)
                {
                    var state = prc.Info();

                    buffer = BitConverter.GetBytes(state.record_count); // 4-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    buffer = BitConverter.GetBytes(state.current_total_accruacy); // 8-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    buffer = new byte[1] { (byte)(state.out_of_line ? 1 : 0) }; // 1-bytes
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

            // serialize out-of-line count
            buffer = BitConverter.GetBytes(instructor.OutOfLine.Count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize out-of-line
            foreach (var bi in instructor.OutOfLine)
            {
                buffer = BitConverter.GetBytes(bi.accuracy); // 8-bytes
                stream.Write(buffer, 0, buffer.Length);

                NeuralNetworkSerializer.Serialize(stream, bi.image);
            }
        }

        public static void Restore(string path, Instructor instructor)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var buffer = new byte[FILE_TYPE_SIGNATURE.Length];
            stream.Read(buffer, 0, buffer.Length);
            var file_type_dignature = Encoding.ASCII.GetString(buffer);

            // check file signature
            bool valid_sign;
            if (file_type_dignature != FILE_TYPE_SIGNATURE_STRING)
            {
                valid_sign = false;
                stream.Seek(0, SeekOrigin.Begin);
            }
            else valid_sign = true;

            // restore file
            Restore(stream, instructor, valid_sign);
        }
        public static void Restore(FileStream stream, Instructor instructor)
        {
            Restore(stream, instructor, true);
        }
        private static void Restore(FileStream stream, Instructor instructor, bool valid_sign)
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
                case 4:
                    RestoreLastVersion4(stream, instructor);
                    break;
                case VERSION:
                    RestoreLastVersion(stream, instructor);
                    break;
                default: throw new Exception("This version of process list is not supported");
            };
        }
        private static void RestoreLastVersion(FileStream stream, Instructor instructor)
        {
            var buffer = new byte[8];

            stream.Read(buffer, 0, 4);
            instructor.Epoch = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            instructor.Offset = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 1);
            instructor.Stage = (TraingingStages)buffer[0];

            stream.Read(buffer, 0, 4);
            var count = BitConverter.ToInt32(buffer, 0);
            var process = new List<TrainProcess>(count);

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 4);
                var record_count = BitConverter.ToInt32(buffer, 0);

                stream.Read(buffer, 0, 8);
                var current_total_accruacy = BitConverter.ToDouble(buffer, 0);

                stream.Read(buffer, 0, 1);
                var is_out_of_line = buffer[0] != 0;

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

                process.Add(TrainProcess.RestoreInfo(
                    new ProgressState(
                        current_image, record_count, current_total_accruacy,
                        accuracy_chain, best_image, is_out_of_line)));
            }

            stream.Read(buffer, 0, 4);
            count = BitConverter.ToInt32(buffer, 0);
            var out_of_line = new List<BrainInfo>(count);

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 8);
                var accruacy = BitConverter.ToDouble(buffer, 0);

                var image = NeuralNetworkSerializer.Restore(stream);

                out_of_line[i] = new BrainInfo(image, accruacy);
            }

            instructor.LoadProgress(process, out_of_line);
        }
        private static void RestoreLastVersion4(FileStream stream, Instructor instructor)
        {
            var buffer = new byte[8];

            stream.Read(buffer, 0, 4);
            var processes_count = BitConverter.ToInt32(buffer, 0);
            var process = new List<TrainProcess>(processes_count);

            for (var i = 0; i < processes_count; i++)
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

                process.Add(TrainProcess.RestoreInfo(
                    new ProgressState(
                        current_image, record_count, current_total_accruacy,
                        accuracy_chain, best_image, false)));
            }

            instructor.LoadProgress(process, null);
        }


        private readonly static byte[] FILE_TYPE_SIGNATURE;
        static TrainProcessSerializer()
        {
            FILE_TYPE_SIGNATURE = Encoding.ASCII.GetBytes(FILE_TYPE_SIGNATURE_STRING);
        }
    }
}
