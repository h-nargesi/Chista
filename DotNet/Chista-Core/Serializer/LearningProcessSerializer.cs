using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Photon.NeuralNetwork.Chista.Trainer;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class LearningProcessSerializer
    {
        public const byte SECTION_TYPE = 2;
        public const ushort VERSION = 9, SECTION_START_SIGNAL = 0xFFFF;
        public const string FILE_TYPE_SIGNATURE_STRING = "Chista Training Process File";

        public static void Serialize(string path, Instructor instructor, string extra)
        {
            Serialize(path, instructor,
                extra == null || extra.Length < 1 ? null :
                Encoding.UTF8.GetBytes(extra));
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
            stream.Write(Encoding.ASCII.GetBytes(FILE_TYPE_SIGNATURE_STRING));
            // end of file signature
            stream.Write(new byte[1], 0, 1);

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

            // new version signal
            buffer = BitConverter.GetBytes(SECTION_START_SIGNAL); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize section type
            buffer = BitConverter.GetBytes(SECTION_TYPE); // 1-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize version
            buffer = BitConverter.GetBytes(VERSION); // 2-bytes
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

            // serialize process
            foreach (NetProcess prc in instructor.Processes)
                NetProcessSerializer.Serialize(stream, prc);

            // serialize out-of-line count
            buffer = BitConverter.GetBytes(instructor.OutOfLines.Count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize out-of-line
            foreach (NetProcess prc in instructor.OutOfLines)
                NetProcessSerializer.Serialize(stream, prc);
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
            var file_type_signature = SectionType.ReadSigniture(stream, Encoding.ASCII);

            if (file_type_signature != FILE_TYPE_SIGNATURE_STRING)
                throw new Exception("Invalid nnp file signature");

            // restore file
            var process = Restore(stream);

            if (stream.Position < stream.Length)
            {
                extra = new byte[stream.Length - stream.Position];
                stream.Read(extra, 0, extra.Length);
            }
            else extra = null;

            return process;
        }
        public static LearningProcessInfo Restore(FileStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");

            // 1: read version: 2-bytes
            var buffer = new byte[2];

            stream.Read(buffer, 0, buffer.Length);
            var signal = BitConverter.ToUInt16(buffer, 0);
            if (signal != SECTION_START_SIGNAL)
                throw new Exception("Invalid section start signal");

            stream.Read(buffer, 0, 1);
            var section_type = buffer[0];
            if (section_type != SECTION_TYPE)
                throw new Exception("Invalid nnp section type");

            stream.Read(buffer, 0, buffer.Length);
            var version = BitConverter.ToUInt16(buffer, 0);

            if (version <= 8)
                throw new Exception("This version of nnp is not supported any more.");

            return version switch
            {
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
            process_info.Processes = new List<INetProcess>(count);

            for (var i = 0; i < count; i++)
                process_info.Processes.Add(NetProcessSerializer.Restor(stream));

            stream.Read(buffer, 0, 4);
            count = BitConverter.ToInt32(buffer, 0);
            process_info.OutOfLine = new List<INetProcess>(count);

            for (var i = 0; i < count; i++)
                process_info.OutOfLine.Add(NetProcessSerializer.Restor(stream));

            return process_info;
        }
    }
}
