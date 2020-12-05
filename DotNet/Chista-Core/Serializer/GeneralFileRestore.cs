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
            var file_type_signature = SectionType.ReadSigniture(stream, Encoding.ASCII);
            object result;

            switch (file_type_signature)
            {
                case LearningProcessSerializer.FILE_TYPE_SIGNATURE_STRING:
                    result = LearningProcessSerializer.Restore(stream);
                    extra = ReadExtra(stream);
                    return result;

                case NeuralNetworkSerializer.FILE_TYPE_SIGNATURE_STRING:
                    result = NeuralNetworkSerializer.Restore(stream);
                    extra = ReadExtra(stream);
                    return result;

                case NeuralNetworkLineSerializer.FILE_TYPE_SIGNATURE_STRING:
                    result = NeuralNetworkLineSerializer.Restore(stream);
                    extra = ReadExtra(stream);
                    return result;

                default:
                    throw new Exception($"Invalid file type ({file_type_signature}).");
            }
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
