using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Photon.NeuralNetwork.Chista.Trainer;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    static class NetProcessSerializer
    {
        public const ushort VERSION = 1;

        public static void Serialize(FileStream stream, NetProcess process)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (process == null)
                throw new ArgumentNullException(nameof(process));

            byte[] buffer;
            // serialize version
            buffer = BitConverter.GetBytes(VERSION); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            var state = process.ProcessInfo();

            buffer = BitConverter.GetBytes(state.running_record_count); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            buffer = BitConverter.GetBytes(state.running_total_accruacy); // 8-bytes
            stream.Write(buffer, 0, buffer.Length);

            buffer = BitConverter.GetBytes(state.accuracy_chain_history.Length); // 4-bytes
            stream.Write(buffer, 0, buffer.Length);

            foreach (var ac in state.accuracy_chain_history)
            {
                buffer = BitConverter.GetBytes(ac); // 8-bytes
                stream.Write(buffer, 0, buffer.Length);
            }

            if (state.running_image is NeuralNetworkImage cu_image)
                NeuralNetworkSerializer.Serialize(stream, cu_image);
            else if (state.running_image is NeuralNetworkLineImage cu_line_image)
                NeuralNetworkLineSerializer.Serialize(stream, cu_line_image);
            else throw new Exception(
                "The serializer for this type of INeuralNetworkImage is not registed");

            if (state.stable_image == null)
            {
                buffer = new byte[] { 0 }; // 1-byte
                stream.Write(buffer, 0, buffer.Length);
            }
            else
            {
                buffer = new byte[] { 1 }; // 1-byte
                stream.Write(buffer, 0, buffer.Length);

                if (state.stable_image is NeuralNetworkImage bs_image)
                    NeuralNetworkSerializer.Serialize(stream, bs_image);
                else if (state.stable_image is NeuralNetworkLineImage bs_line_image)
                    NeuralNetworkLineSerializer.Serialize(stream, bs_line_image);
                else throw new Exception(
                    "The serializer for this type of INeuralNetworkImage is not registed");
            }
        }

        public static NetProcess Restor(FileStream stream)
        {
            var buffer = new byte[8];

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

            return new NetProcess(
                new NetProcessInfo(
                    current_image, record_count, current_total_accruacy,
                    best_image, accuracy_chain));
        }
    }
}
