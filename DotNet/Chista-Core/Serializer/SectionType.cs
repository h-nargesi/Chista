using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    class SectionType
    {
        public const ushort FILE_TYPE_MASK = 0XF000, VERSION_MASK = 0xFFF;

        public static ushort GetSectionSign(byte file_type, ushort version)
        {
            return (ushort)((file_type << 12) | version);
        }
        public static (byte file_type, ushort version) GetSectionInfo(ushort file_sign)
        {
            return ((byte)((FILE_TYPE_MASK & file_sign) >> 12), (ushort)(VERSION_MASK & file_sign));
        }

        public static string ReadSigniture(FileStream stream, Encoding encoding)
        {
            int data;
            var buffer = new List<byte>();
            do
            {
                data = stream.ReadByte();
                buffer.Add((byte)data);
            }
            while (data > 0);

            return encoding.GetString(buffer.ToArray());
        }
    }
}
