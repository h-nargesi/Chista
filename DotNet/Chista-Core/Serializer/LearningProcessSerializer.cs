using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Photon.NeuralNetwork.Chista.Trainer;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class LearningProcessSerializer
    {
        public const string FILE_TYPE_SIGNATURE_STRING = "Opertat Training Process File";
        public const byte SECTION_TYPE = 2;
        public const ushort VERSION = 8;

        public static void Serialize(string path, Instructor instructor, string extra)
        {
            Serialize(path, instructor, extra == null || extra.Length < 1 ? null : Encoding.UTF8.GetBytes(extra));
        }
        public static void Serialize(string path, Instructor instructor, byte[] extra = null)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (instructor == null)
                throw new ArgumentNullException(nameof(instructor));

            using var stream = File.Create(path);
            stream.Flush();

            // serialize file signature
            stream.Write(FILE_TYPE_SIGNATURE);

            // serialize main data
            Serialize(stream, instructor);

            // serialize developer extra data
            if (extra != null && extra.Length > 0)
                stream.Write(extra);
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

            // process's stage info
            // serialize process stage
            buffer = new byte[1] { (byte)instructor.Stage }; // 1-bytes
            stream.Write(buffer, 0, buffer.Length);
            // serialize process offset
            buffer = BitConverter.GetBytes(instructor.Offset); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);
            // serialize process epoch
            buffer = BitConverter.GetBytes(instructor.Epoch); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize process count
            buffer = BitConverter.GetBytes(instructor.Processes.Count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // functions serializer
            var function = new FunctionSerializerCore(stream);

            // serialize process
            foreach (var iprc in instructor.Processes)
                if (iprc is TrainingProcess prc)
                {
                    var state = prc.ProcessInfo();

                    buffer = BitConverter.GetBytes(state.record_count); // 4-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    buffer = BitConverter.GetBytes(state.current_total_accruacy); // 8-bytes
                    stream.Write(buffer, 0, buffer.Length);

                    function.Serialize(prc.Accurate);

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
                buffer = BitConverter.GetBytes(bi.Accuracy); // 8-bytes
                stream.Write(buffer, 0, buffer.Length);

                function.Serialize(bi.Accurate);

                NeuralNetworkSerializer.Serialize(stream, bi.Image);
            }
        }

        public static LearningProcessInfo Restore(string path)
        {
            return Restore(path, out byte[] _);
        }
        public static LearningProcessInfo Restore(string path, out string extra)
        {
            var prc = Restore(path, out byte[] extra_bytes);
            if (extra_bytes == null) extra = null;
            else extra = Encoding.UTF8.GetString(extra_bytes);
            return prc;
        }
        public static LearningProcessInfo Restore(string path, out byte[] extra)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var buffer = new byte[FILE_TYPE_SIGNATURE.Length];
            stream.Read(buffer, 0, buffer.Length);
            var file_type_dignature = Encoding.ASCII.GetString(buffer);

            if (file_type_dignature != FILE_TYPE_SIGNATURE_STRING)
                throw new Exception("Invalid nnp file signature");

            // restore file
            var prc = Restore(stream);

            if (stream.Position < stream.Length)
            {
                extra = new byte[stream.Length - stream.Position];
                stream.Read(extra, 0, extra.Length);
            }
            else extra = null;

            return prc;
        }
        public static LearningProcessInfo Restore(FileStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");

            // 1: read version: 2-bytes
            var buffer = new byte[2];
            stream.Read(buffer, 0, buffer.Length);
            var (section_type, version) = SectionType.GetSectionInfo(BitConverter.ToUInt16(buffer, 0));

            if (section_type != SECTION_TYPE && version > 2)
                throw new Exception("Invalid nnp section type");

            if (version <= 5)
                throw new Exception("This version of nnp is not supported any more.");

            return version switch
            {
                6 => RestoreLastVersion6(stream),
                7 => RestoreLastVersion7(stream),
                VERSION => RestoreLastVersion(stream),
                _ => throw new Exception("This version of nnp list is not supported"),
            };
        }
        private static LearningProcessInfo RestoreLastVersion(FileStream stream)
        {
            var process_info = new LearningProcessInfo();
            var buffer = new byte[8];

            stream.Read(buffer, 0, 1);
            process_info.Stage = (TrainingStages)buffer[0];

            stream.Read(buffer, 0, 4);
            process_info.Offset = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            process_info.Epoch = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            var count = BitConverter.ToInt32(buffer, 0);
            process_info.Processes = new List<ITrainingProcess>(count);

            var function = new FunctionSerializerCore(stream);

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 4);
                var record_count = BitConverter.ToInt32(buffer, 0);

                stream.Read(buffer, 0, 8);
                var current_total_accruacy = BitConverter.ToDouble(buffer, 0);

                var accurate = function.RestoreIAccurateGauge();

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

                process_info.Processes.Add(new TrainingProcess(
                    new NetProcessInfo(
                        current_image, record_count, current_total_accruacy,
                        accuracy_chain, best_image, accurate)));
            }

            stream.Read(buffer, 0, 4);
            count = BitConverter.ToInt32(buffer, 0);
            process_info.OutOfLine = new List<IBrainInfo>(count);

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 8);
                var accruacy = BitConverter.ToDouble(buffer, 0);
                var accurate = function.RestoreIAccurateGauge();
                var image = NeuralNetworkSerializer.Restore(stream);

                process_info.OutOfLine.Add(new BrainInfo(image, accruacy, accurate));
            }

            return process_info;
        }
        private static LearningProcessInfo RestoreLastVersion7(FileStream stream)
        {
            var process_info = new LearningProcessInfo();
            var buffer = new byte[8];

            stream.Read(buffer, 0, 1);
            process_info.Stage = (TrainingStages)buffer[0];

            stream.Read(buffer, 0, 4);
            process_info.Offset = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            process_info.Epoch = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            var count = BitConverter.ToInt32(buffer, 0);
            process_info.Processes = new List<ITrainingProcess>(count);

            var default_accurat = new AccurateGauge();

            for (var i = 0; i < count; i++)
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

                process_info.Processes.Add(new TrainingProcess(
                    new NetProcessInfo(
                        current_image, record_count, current_total_accruacy,
                        accuracy_chain, best_image, default_accurat)));
            }

            stream.Read(buffer, 0, 4);
            count = BitConverter.ToInt32(buffer, 0);
            process_info.OutOfLine = new List<IBrainInfo>(count);

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 8);
                var accruacy = BitConverter.ToDouble(buffer, 0);

                var image = NeuralNetworkSerializer.Restore(stream);

                process_info.OutOfLine.Add(new BrainInfo(image, accruacy, default_accurat));
            }

            return process_info;
        }
        private static LearningProcessInfo RestoreLastVersion6(FileStream stream)
        {
            var process_info = new LearningProcessInfo();
            var buffer = new byte[8];

            stream.Read(buffer, 0, 1);
            process_info.Stage = (TrainingStages)buffer[0];

            stream.Read(buffer, 0, 4);
            process_info.Offset = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            process_info.Epoch = BitConverter.ToUInt32(buffer, 0);

            stream.Read(buffer, 0, 4);
            var count = BitConverter.ToInt32(buffer, 0);
            process_info.Processes = new List<ITrainingProcess>(count);

            var default_accurat = new AccurateGauge();

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 4);
                var record_count = BitConverter.ToInt32(buffer, 0);

                stream.Read(buffer, 0, 8);
                var current_total_accruacy = BitConverter.ToDouble(buffer, 0);

                // out of line is deprecated
                stream.Read(buffer, 0, 1);

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

                process_info.Processes.Add(new TrainingProcess(
                    new NetProcessInfo(
                        current_image, record_count, current_total_accruacy,
                        accuracy_chain, best_image, default_accurat)));
            }

            stream.Read(buffer, 0, 4);
            count = BitConverter.ToInt32(buffer, 0);
            process_info.OutOfLine = new List<IBrainInfo>(count);

            for (var i = 0; i < count; i++)
            {
                stream.Read(buffer, 0, 8);
                var accruacy = BitConverter.ToDouble(buffer, 0);

                var image = NeuralNetworkSerializer.Restore(stream);

                process_info.OutOfLine.Add(new BrainInfo(image, accruacy, default_accurat));
            }

            return process_info;
        }


        public static int SIGNATURE_LENGTH => FILE_TYPE_SIGNATURE.Length;
        private readonly static byte[] FILE_TYPE_SIGNATURE;
        static LearningProcessSerializer()
        {
            FILE_TYPE_SIGNATURE = Encoding.ASCII.GetBytes(FILE_TYPE_SIGNATURE_STRING);
        }
    }
}
