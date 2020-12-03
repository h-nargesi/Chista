using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class NeuralNetworkLineSerializer
    {
        public const byte SECTION_TYPE = 3;
        public const ushort VERSION = 1, SECTION_START_SIGNAL = 0xFFFF;
        public const string FILE_TYPE_SIGNATURE_STRING = "Chista Neural Network Line Image";

        public static void Serialize(string path, NeuralNetworkLineImage image)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            using var stream = File.Create(path);
            stream.Flush();

            // serialize file signature
            stream.Write(Encoding.ASCII.GetBytes(FILE_TYPE_SIGNATURE_STRING));
            // end of file signature
            stream.Write(new byte[1], 0, 1);

            Serialize(stream, image);
        }
        public static void Serialize(FileStream stream, NeuralNetworkLineImage image)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

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

            // serialize index
            buffer = BitConverter.GetBytes(image.index); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // functions serializer
            var function = new FunctionSerializerCore(stream);

            // serialize images count
            buffer = BitConverter.GetBytes(image.images.Length); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            // serialize line
            NeuralNetworkSerializer.Serialize(stream, image.images[0]);
            for (int i = 0; i < image.combiners.Length;)
            {
                function.Serialize(image.combiners[i++]);
                NeuralNetworkSerializer.Serialize(stream, image.images[0]);
            }
        }

        public static NeuralNetworkLineImage Restore(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var file_type_signature = SectionType.ReadSigniture(stream, Encoding.ASCII);
            if (file_type_signature != FILE_TYPE_SIGNATURE_STRING)
                throw new Exception("Invalid nnli file signature");

            // restore file
            return Restore(stream);
        }
        public static NeuralNetworkLineImage Restore(FileStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            // 1: read version: 2-bytes
            var buffer = new byte[2];

            stream.Read(buffer, 0, buffer.Length);
            var signal = BitConverter.ToUInt16(buffer, 0);
            if (signal != SECTION_START_SIGNAL)
                throw new Exception("Invalid section start signal");

            stream.Read(buffer, 0, 1);
            var section_type = buffer[0];
            if (section_type != SECTION_TYPE)
                throw new Exception("Invalid nnli section type");

            stream.Read(buffer, 0, buffer.Length);
            var version = BitConverter.ToUInt16(buffer, 0);

            return version switch
            {
                VERSION => RestoreLastVersion(stream),
                _ => throw new Exception("This version of nni is not supported."),
            };
        }
        private static NeuralNetworkLineImage RestoreLastVersion(FileStream stream)
        {
            var buffer = new byte[4];

            stream.Read(buffer, 0, buffer.Length);
            var index = BitConverter.ToUInt16(buffer, 0);

            stream.Read(buffer, 0, buffer.Length);
            var length = BitConverter.ToUInt16(buffer, 0);

            if (length < 1) throw new Exception("Invalid length of images.");

            var function = new FunctionSerializerCore(stream);

            var images = new NeuralNetworkImage[length];
            var combiners = new IDataCombiner[length - 1];

            images[0] = NeuralNetworkSerializer.Restore(stream);
            for (int i = 0; i < combiners.Length;)
            {
                combiners[i++] = function.RestoreIDataCombiner();
                images[i] = NeuralNetworkSerializer.Restore(stream);
            }

            return new NeuralNetworkLineImage(images, combiners, index);
        }
    }
}