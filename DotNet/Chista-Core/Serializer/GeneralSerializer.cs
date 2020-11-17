using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class GeneralSerializer
    {
        public static object Restore(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var buffer = new byte[TrainProcessSerializer.SIGNATURE_LENGTH];
            stream.Read(buffer, 0, buffer.Length);
            var file_type_diignature = Encoding.ASCII.GetString(buffer);

            if (file_type_diignature == TrainProcessSerializer.FILE_TYPE_SIGNATURE_STRING)
                return TrainProcessSerializer.Restore(stream);
            else stream.Seek(0, SeekOrigin.Begin);

            buffer = new byte[NeuralNetworkSerializer.SIGNATURE_LENGTH];
            stream.Read(buffer, 0, buffer.Length);
            file_type_diignature = Encoding.ASCII.GetString(buffer);

            if (file_type_diignature == NeuralNetworkSerializer.FILE_TYPE_SIGNATURE_STRING)
                return NeuralNetworkSerializer.Restore(stream);
            else stream.Seek(0, SeekOrigin.Begin);

            throw new Exception("Invalid file type");
        }
    }
}
