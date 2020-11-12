using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Photon.NeuralNetwork.Opertat.Serializer
{
    public static class NeuralNetworkSerializer
    {
        public const string FILE_TYPE_SIGNATURE_STRING = "Opertat Neural Network Image";
        public const byte SECTION_TYPE = 1;
        public const ushort VERSION = 5;

        public static void Serialize(string path, NeuralNetworkImage image)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            using var stream = File.Create(path);
            stream.Flush();

            // serialize file signature
            stream.Write(FILE_TYPE_SIGNATURE);

            Serialize(stream, image);
        }
        public static void Serialize(FileStream stream, NeuralNetworkImage image)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            // 1: serialize version
            var buffer = BitConverter.GetBytes(SectionType.GetSectionSign(SECTION_TYPE, VERSION)); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // 2: serializer layers
            LayerSerializer.Serialize(stream, image.layers);

            // functions serializer
            var function = new FunctionSerializer(stream);

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
            var buffer = new byte[FILE_TYPE_SIGNATURE.Length];
            stream.Read(buffer, 0, buffer.Length);
            var file_type_diignature = Encoding.ASCII.GetString(buffer);

            if (file_type_diignature != FILE_TYPE_SIGNATURE_STRING)
                throw new Exception("Invalid nni file signature");

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
            var (section_type, version) = SectionType.GetSectionInfo(BitConverter.ToUInt16(buffer, 0));

            if (section_type != SECTION_TYPE && version > 3)
                throw new Exception("Invalid nni section type");

            if(version <= 4)
                throw new Exception("This version of nni is not supported any more.");

            switch (version)
            {
                case VERSION:
                    return RestoreLastVersion(stream);
                default: throw new Exception("This version of nni is not supported.");
            };
        }
        private static NeuralNetworkImage RestoreLastVersion(FileStream stream)
        {
            // 2: read all layers
            var layers = LayerSerializer.Restore(stream);

            // functions serializer
            var function = new FunctionSerializer(stream);

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
        private static NeuralNetworkImage RestoreVer2(FileStream stream)
        {
            // 2: read all layers
            var layers = LayerSerializer.Restore(stream);

            // functions serializer
            var function = new FunctionSerializer(stream);

            // 3: read error function
            var error = function.RestoreIErrorFunction();

            // regularization not supported in this version

            // 4: read data intput convertor
            var input_convertor = function.RestoreIDataConvertor();

            // 5: read data output convertor
            var output_convertor = function.RestoreIDataConvertor();

            return new NeuralNetworkImage(layers, error, input_convertor, output_convertor, null);
        }


        private readonly static byte[] FILE_TYPE_SIGNATURE;
        static NeuralNetworkSerializer()
        {
            FILE_TYPE_SIGNATURE = Encoding.ASCII.GetBytes(FILE_TYPE_SIGNATURE_STRING);
        }
    }
}