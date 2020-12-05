using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class NeuralNetworkGeneralSerializer
    {
        public const byte SECTION_TYPE = 1;
        public const ushort VERSION = 5;

        public static void Serialize(FileStream stream, INeuralNetworkImage image)
        {
            if (image is NeuralNetworkImage cu_image)
                NeuralNetworkSerializer.Serialize(stream, cu_image);
            else if (image is NeuralNetworkLineImage cu_line_image)
                NeuralNetworkLineSerializer.Serialize(stream, cu_line_image);
            else throw new Exception(
                "The serializer for this type of INeuralNetworkImage is not registed.");
        }
        public static INeuralNetworkImage Restore(FileStream stream)
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
            stream.Seek(-3, SeekOrigin.Current);

            return buffer[0] switch
            {
                NeuralNetworkSerializer.SECTION_TYPE => NeuralNetworkSerializer.Restore(stream),
                NeuralNetworkLineSerializer.SECTION_TYPE => NeuralNetworkLineSerializer.Restore(stream),
                _ => throw new Exception($"Invalid neural network section type ({buffer[0]}).")
            };
        }
    }
}
