using System;
using System.IO;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class NeuralNetworkSerializer
    {
        public const ushort VERSION = 2;
        public static void Serialize(NeuralNetworkImage image, string path)
        {
            using var stream = File.Create(path);
            stream.Flush();

            // 1: serialize version
            var buffer = BitConverter.GetBytes(VERSION); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // 2: serializer layers
            LayerSerializer.Serialize(stream, image.layers);

            // functions serializer
            var function = new FunctionSerializer(stream);

            // 3: serialize error function
            function.Serialize(image.error_fnc);

            // 4: serialize data input convertor
            function.Serialize(image.input_convertor);

            // 5: serialize data output convertor
            function.Serialize(image.output_convertor);
        }
        public static NeuralNetworkImage Restore(string path)
        {
            using var stream = File.OpenRead(path);

            // 1: read version: 2-bytes
            var buffer = new byte[2];
            stream.Read(buffer, 0, buffer.Length);
            var version = BitConverter.ToUInt16(buffer, 0);
            CheckVersion(version);

            // 2: read all layers
            var layers = LayerSerializer.Restore(stream, version);

            // functions serializer
            var function = new FunctionSerializer(stream);

            // 3: read error function
            var error = function.RestorIErrorFunction();

            // 4: read data intput convertor
            var input_convertor = function.RestorIDataConvertor();

            // 5: read data output convertor
            var output_convertor = function.RestorIDataConvertor();

            return new NeuralNetworkImage(layers, error, input_convertor, output_convertor);
        }
        private static void CheckVersion(ushort version)
        {
            switch (version)
            {
                case VERSION:
                    return;
                default:
                    throw new Exception("This version of nni is not supported");
            }
        }

    }
}