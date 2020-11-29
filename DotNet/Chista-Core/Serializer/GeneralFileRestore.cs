using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public static class GeneralFileRestore
    {
        public static object Restore(string path)
        {
            return Restore(path, out byte[] _);
        }
        public static object Restore(string path, out string extra)
        {
            var obj = Restore(path, out byte[] extra_bytes);
            if (extra_bytes == null) extra = null;
            else extra = Encoding.UTF8.GetString(extra_bytes);
            return obj;
        }
        public static object Restore(string path, out byte[] extra)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var stream = File.OpenRead(path);

            // read file signature
            var buffer = new byte[TrainProcessSerializer.SIGNATURE_LENGTH];
            stream.Read(buffer, 0, buffer.Length);
            var file_type_diignature = Encoding.ASCII.GetString(buffer);

            if (file_type_diignature == TrainProcessSerializer.FILE_TYPE_SIGNATURE_STRING)
            {
                var prc = TrainProcessSerializer.Restore(stream);
                extra = ReadExtra(stream);
                return prc;
            }
            else stream.Seek(0, SeekOrigin.Begin);

            buffer = new byte[NeuralNetworkSerializer.SIGNATURE_LENGTH];
            stream.Read(buffer, 0, buffer.Length);
            file_type_diignature = Encoding.ASCII.GetString(buffer);

            if (file_type_diignature == NeuralNetworkSerializer.FILE_TYPE_SIGNATURE_STRING)
            {
                var image = NeuralNetworkSerializer.Restore(stream);
                extra = ReadExtra(stream);
                return image;
            }
            else stream.Seek(0, SeekOrigin.Begin);

            throw new Exception("Invalid file type");
        }
        private static byte[] ReadExtra(FileStream stream)
        {
            if (stream.Position < stream.Length)
            {
                var extra = new byte[stream.Length - stream.Position];
                stream.Read(extra, 0, extra.Length);
                return extra;
            }
            else return null;
        }
    }
}
