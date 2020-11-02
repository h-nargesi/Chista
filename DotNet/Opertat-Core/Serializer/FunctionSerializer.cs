using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

namespace Photon.NeuralNetwork.Opertat.Implement
{
    class FunctionSerializer
    {
        private readonly FileStream stream;
        public FunctionSerializer(FileStream stream)
        {
            this.stream = stream ??
                throw new ArgumentNullException(nameof(stream), "The writer stream is not defined");
        }

        public void Serialize(IErrorFunction error_func)
        {
            ushort code;
            List<byte> parameters;
            switch (error_func)
            {
                case Errorest _:
                    code = 1;
                    parameters = null;
                    break;

                case ErrorStack e:
                    code = 2;
                    parameters = new List<byte>();
                    parameters.AddRange(BitConverter.GetBytes(e.IndexCount));
                    break;

                default:
                    throw new ArgumentException(
                        nameof(error_func), "this type of IErrorFunction is not registered.");
            }

            // serialaize type and parameters
            Serialize(code, parameters?.ToArray());
        }
        public void Serialize(IDataConvertor convertor)
        {
            ushort code;
            List<byte> parameters;
            if (convertor == null)
            {
                code = 0;
                parameters = null;
            }
            else switch (convertor)
                {
                    case DataRange data_range:
                        code = 1;
                        parameters = new List<byte>();
                        parameters.AddRange(BitConverter.GetBytes(data_range.SignalRange));
                        parameters.AddRange(BitConverter.GetBytes(data_range.SignalHeight));
                        break;
                    default:
                        throw new ArgumentException(
                            nameof(convertor), "this type of IDataConvertor is not registered.");
                }

            // serialaize type and parameters
            Serialize(code, parameters?.ToArray());
        }
        public void Serialize(IRegularization regularization)
        {
            ushort code;
            List<byte> parameters;
            if (regularization == null)
            {
                code = 0;
                parameters = null;
            }
            else switch (regularization)
                {
                    case RegularizationL1 _:
                        code = 1;
                        parameters = null;
                        break;
                    case RegularizationL2 _:
                        code = 2;
                        parameters = null;
                        break;
                    default:
                        throw new ArgumentException(
                            nameof(regularization), "this type of IRegularization is not registered.");
                }

            // serialaize type and parameters
            Serialize(code, parameters?.ToArray());
        }
        private void Serialize(ushort code, byte[] parameters)
        {
            // serialaize type
            var buffer = BitConverter.GetBytes(code); // 2-bytes
            stream.Write(buffer, 0, buffer.Length);

            // without parameters
            if (parameters == null || parameters.Length == 0) return;

            // serialaize parameters
            stream.Write(parameters, 0, parameters.Length);
        }

        public IErrorFunction RestoreIErrorFunction()
        {
            byte[] buffer;
            var code = RestorFunctionType();

            switch (code)
            {
                case 1: return new Errorest();
                case 2:
                    buffer = RestorParameter(4);
                    return new ErrorStack(BitConverter.ToInt32(buffer, 0));
                default:
                    throw new Exception(
                        $"this type of IErrorFunction ({code}) is not registered.");
            }
        }
        public IDataConvertor RestoreIDataConvertor()
        {
            byte[] buffer;
            var code = RestorFunctionType();

            switch (code)
            {
                case 0: return null;
                case 1:
                    buffer = RestorParameter(8);
                    return new DataRange(
                        BitConverter.ToUInt32(buffer, 0),
                        BitConverter.ToInt32(buffer, 4));
                default:
                    throw new Exception(
                        $"This type of IDataConvertor ({code}) is not registered.");
            }
        }
        public IRegularization RestoreIRegularization()
        {
            var code = RestorFunctionType();

            switch (code)
            {
                case 0: return null;
                case 1: return new RegularizationL1();
                case 2: return new RegularizationL2();
                default:
                    throw new Exception(
                        $"this type of IRegularization ({code}) is not registered.");
            }
        }
        private ushort RestorFunctionType()
        {
            var buffer = new byte[2];
            stream.Read(buffer, 0, buffer.Length);
            return BitConverter.ToUInt16(buffer, 0);
        }
        private byte[] RestorParameter(int length)
        {
            var buffer = new byte[length];
            stream.Read(buffer, 0, buffer.Length);
            return buffer;
        }

        private static readonly Dictionary<ushort, IConduction> all_conductions =
            new Dictionary<ushort, IConduction>();

        public static ushort EnCodeIConduction(IConduction conduction)
        {
            return conduction switch
            {
                Sigmoind _ => 1,
                ReLU _ => 2,
                SoftReLU _ => 3,
                Straight _ => 4,
                _ => throw new ArgumentException(
                    nameof(conduction), "this type of IConduction is not registered."),
            };
        }
        [MethodImpl(MethodImplOptions.Synchronized)]
        public static IConduction DecodeIConduction(ushort value)
        {
            if (all_conductions.ContainsKey(value))
                return all_conductions[value];
            IConduction conduction = value switch
            {
                1 => new Sigmoind(),
                2 => new ReLU(),
                3 => new SoftReLU(),
                4 => new Straight(),
                _ => throw new ArgumentException(
                    nameof(value), "this type of IConduction is not registered."),
            };
            all_conductions.Add(value, conduction);
            return conduction;
        }

    }
}