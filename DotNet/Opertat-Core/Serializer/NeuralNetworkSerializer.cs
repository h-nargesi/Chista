using System;
using System.IO;

namespace Photon.NeuralNetwork.Opertat.Serializer
{
    public static class NeuralNetworkSerializer
    {
        public const ushort VERSION = 3;

        public static void Serialize(string path, NeuralNetworkImage image)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            using var stream = File.Create(path);
            stream.Flush();
            Serialize(stream, image);
        }
        public static void Serialize(FileStream stream, NeuralNetworkImage image)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            // 1: serialize version
            var buffer = BitConverter.GetBytes(VERSION); // 2-bytes
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
            return Restore(stream);
        }
        public static NeuralNetworkImage Restore(FileStream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            // 1: read version: 2-bytes
            var buffer = new byte[2];
            stream.Read(buffer, 0, buffer.Length);
            var version = BitConverter.ToUInt16(buffer, 0);

            return version switch
            {
                1 => throw new Exception("The 1st version of nni is not supported any more."),
                2 => RestoreVer2(stream),
                VERSION => RestoreLastVersion(stream),
                _ => throw new Exception("This version of nni is not supported."),
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
    }
}