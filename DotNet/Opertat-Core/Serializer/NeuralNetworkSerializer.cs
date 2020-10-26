using System;
using System.IO;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class NeuralNetworkSerializer
    {
        public const ushort VERSION = 3;
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

            // 4: serialize regularization function
            function.Serialize(image.regularization);

            // 5: serialize data input convertor
            function.Serialize(image.input_convertor);

            // 6: serialize data output convertor
            function.Serialize(image.output_convertor);
        }
        public static NeuralNetworkImage Restore(string path)
        {
            using var stream = File.OpenRead(path);

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

            // 4: read regulazor function
            var regulazation = function.RestoreIRegularization();

            // 5: read data intput convertor
            var input_convertor = function.RestoreIDataConvertor();

            // 6: read data output convertor
            var output_convertor = function.RestoreIDataConvertor();

            return new NeuralNetworkImage(layers, error, input_convertor, output_convertor, regulazation);
        }
        private static NeuralNetworkImage RestoreVer2(FileStream stream)
        {
            // 2: read all layers
            var layers = LayerSerializer.Restore(stream);

            // functions serializer
            var function = new FunctionSerializer(stream);

            // 3: read error function
            var error = function.RestoreIErrorFunction();

            // regulazation not supported in this version

            // 4: read data intput convertor
            var input_convertor = function.RestoreIDataConvertor();

            // 5: read data output convertor
            var output_convertor = function.RestoreIDataConvertor();

            return new NeuralNetworkImage(layers, error, input_convertor, output_convertor, null);
        }

    }
}