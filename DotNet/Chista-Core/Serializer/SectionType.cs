using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    class SectionType
    {
        public const ushort SECTION_START_SIGNAL = 0xFFFF,
            FILE_TYPE_MASK = 0XF000, VERSION_MASK = 0xFFF;

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

            while ((data = stream.ReadByte()) > 0)
                buffer.Add((byte)data);

            return encoding.GetString(buffer.ToArray());
        }
    }
}
