using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Opertat
{
    public static class Convertor
    {
        public static void BinaryState(byte value, double[] buffer, ref int index)
        {
            var mask = 1;
            var max_index = index + 8;
            for (; index < max_index; mask *= 2, index++)
                buffer[index] = (value & mask) == 0 ? 0D : 1D;
        }
        public static void BinaryState(byte[] values, double[] buffer, ref int index)
        {
            for (int i = 0; i < values.Length; i++)
                BinaryState(values[i], buffer, ref index);
        }
        public static void BinaryState(sbyte value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }
        public static void BinaryState(short value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }
        public static void BinaryState(ushort value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }
        public static void BinaryState(int value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }
        public static void BinaryState(uint value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }
        public static void BinaryState(long value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }
        public static void BinaryState(ulong value, double[] buffer, ref int index)
        {
            BinaryState(BitConverter.GetBytes(value), buffer, ref index);
        }

    }
}
