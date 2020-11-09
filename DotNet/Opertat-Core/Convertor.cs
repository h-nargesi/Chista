using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Opertat
{
    public static class Convertor
    {
        public static void BinaryState(sbyte value, double[] buffer, ref int index)
        {
            var mask = 1;
            for (int i = 0; i < 8; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(byte value, double[] buffer, ref int index)
        {
            var mask = 1;
            for (int i = 0; i < 8; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(short value, double[] buffer, ref int index)
        {
            var mask = 1;
            for (int i = 0; i < 16; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(ushort value, double[] buffer, ref int index)
        {
            var mask = 1;
            for (int i = 0; i < 16; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(int value, double[] buffer, ref int index)
        {
            var mask = 1;
            for (int i = 0; i < 32; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(uint value, double[] buffer, ref int index)
        {
            var mask = 1U;
            for (int i = 0; i < 64; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(long value, double[] buffer, ref int index)
        {
            var mask = 1L;
            for (int i = 0; i < 32; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }
        public static void BinaryState(ulong value, double[] buffer, ref int index)
        {
            var mask = 1UL;
            for (int i = 0; i < 64; i++, mask *= 2, index++)
                buffer[index] = (value & mask) == 1 ? 1D : 0D;
        }

        public static void BinaryState(byte[] values, double[] buffer, ref int index)
        {
            for (int i = 0; i < values.Length; i++)
                BinaryState(values[i], buffer, ref index);
        }
    }
}
