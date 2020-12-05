using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class NeuralNetworkSerializer
    {
        public const byte SECTION_TYPE = 1;
        public const ushort VERSION = 5;
        public const string FILE_TYPE_SIGNATURE_STRING = "Chista Neural Network Image";

        public static void Serialize(string path, NeuralNetworkImage image)
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
        public static void Serialize(FileStream stream, NeuralNetworkImage image)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            byte[] buffer;

            // 1: new version signal
            buffer = BitConverter.GetBytes(SectionType.SECTION_START_SIGNAL); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // 2: serialize section type
            buffer = new byte[] { SECTION_TYPE }; // 1-bytes
            stream.Write(buffer, 0, buffer.Length);

            // 3: serialize version
            buffer = BitConverter.GetBytes(VERSION); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // 2: serializer layers
            LayerSerializer.Serialize(stream, image.layers);

            // functions serializer
            var function = new FunctionSerializerCore(stream);

            // 3: serialize error function
            function.Serialize(image.error_fnc);

            // 4: serialize regularization function
            function.Serialize(image.regularization);

            // 5: serialize data input convertor
            function.Serialize(image.input_convertor);

            // 6: serialize data output convertor
            function.Serialize(image.output_convertor);
        }

        public static NeuralNetworkImage Restore(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var file_type_signature = SectionType.ReadSigniture(stream, Encoding.ASCII);
            if (file_type_signature != FILE_TYPE_SIGNATURE_STRING)
                throw new Exception($"Invalid nni file signature ({file_type_signature}).");

            // restore file
            return Restore(stream);

        }
        public static NeuralNetworkImage Restore(FileStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            // 1: read version: 2-bytes
            var buffer = new byte[2];

            stream.Read(buffer, 0, buffer.Length);
            var signal = BitConverter.ToUInt16(buffer, 0);
            if (signal != SectionType.SECTION_START_SIGNAL)
                throw new Exception($"Invalid section start signal ({signal}).");

            stream.Read(buffer, 0, 1);
            var section_type = buffer[0];
            if (section_type != SECTION_TYPE)
                throw new Exception($"Invalid nnli section type ({section_type}).");

            stream.Read(buffer, 0, buffer.Length);
            var version = BitConverter.ToUInt16(buffer, 0);

            if (version <= 4)
                throw new Exception($"This version ({version}) of nni is not supported any more.");

            return version switch
            {
                VERSION => RestoreLastVersion(stream),
                _ => throw new Exception($"This version ({version}) of nni is not supported."),
            };
        }
        private static NeuralNetworkImage RestoreLastVersion(FileStream stream)
        {
            // 2: read all layers
            var layers = LayerSerializer.Restore(stream);

            // functions serializer
            var function = new FunctionSerializerCore(stream);

            // 3: read error function
            var error = function.RestoreIErrorFunction();

            // 4: read regularizer function
            var regularization = function.RestoreIRegularization();

            // 5: read data intput convertor
            var input_convertor = function.RestoreIDataConvertor();

            // 6: read data output convertor
            var output_convertor = function.RestoreIDataConvertor();

            return new NeuralNetworkImage(layers, error, input_convertor, output_convertor, regularization);
        }

    }
}