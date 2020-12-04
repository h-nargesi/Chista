using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using Photon.NeuralNetwork.Chista.Implement;
using Photon.NeuralNetwork.Chista.Trainer;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    class FunctionSerializerCore
    {
        public const int
            ERROR_FUNCTION_SIGN = 0x10000,
            DATA_CONVERTOR_SIGN = 0x20000,
            REGULARIZATION_SIGN = 0x40000,
            ACCURATE_GAUGE_SIGN = 0x80000,
            DATA_COMBINER_SIGN = 0x100000;
        private readonly static Dictionary<int, IFunctionSerializer> registered_functions_via_code =
            new Dictionary<int, IFunctionSerializer>();
        public static void RegisterFunction<T>(FunctionSerializer<T> serializer)
            where T : ISerializableFunction
        {
            if (registered_functions_via_code.ContainsKey(serializer.Ucode))
                throw new ArgumentException(nameof(serializer),
                    $"Duplicated Code({serializer.Code}, {serializer.Ucode}). This code is registred for {registered_functions_via_code[serializer.Ucode].Name}.");

            registered_functions_via_code.Add(serializer.Ucode, serializer);
        }
        public static FunctionCode GetAttribute(Type type)
        {
            var attr = type.GetCustomAttribute<FunctionCode>();
            if (attr == null)
                throw new ArgumentException(type.Name,
                    $"The type of serializable function ({type.Name}) is not have 'FunctionCode' attribute.");
            return attr;
        }

        private readonly FileStream stream;
        public FunctionSerializerCore(FileStream stream)
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

                case Deprecated.ErrorStack e:
                    code = 2;
                    parameters = new List<byte>();
                    parameters.AddRange(BitConverter.GetBytes(e.IndexCount));
                    break;

                case Classification e:
                    code = 3;
                    parameters = new List<byte>();
                    parameters.AddRange(BitConverter.GetBytes(e.MinAccept));
                    break;

                case Tagging e:
                    code = 4;
                    parameters = new List<byte>();
                    parameters.AddRange(BitConverter.GetBytes(e.MinAccept));
                    parameters.AddRange(BitConverter.GetBytes(e.MaxReject));
                    break;

                default:
                    var attr = GetAttribute(error_func.GetType());
                    if (!registered_functions_via_code.ContainsKey(attr.code | ERROR_FUNCTION_SIGN))
                        throw new ArgumentException(
                            nameof(error_func), "this type of IErrorFunction is not registered.");
                    var serializer = registered_functions_via_code[attr.code | ERROR_FUNCTION_SIGN];
                    code = serializer.Code;
                    parameters = new List<byte>();
                    parameters.AddRange(serializer.Serialize(error_func) ?? new byte[0]);
                    if (parameters.Count != serializer.ParameterLength)
                        throw new Exception("invalid parameters' length.");
                    break;
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
                        // paramter length = 8
                        break;

                    case DataRangeDouble data_range_double:
                        code = 2;
                        parameters = new List<byte>();
                        parameters.AddRange(BitConverter.GetBytes(data_range_double.SignalRange));
                        parameters.AddRange(BitConverter.GetBytes(data_range_double.SignalHeight));
                        // paramter length = 16
                        break;

                    default:
                        var attr = GetAttribute(convertor.GetType());
                        if (!registered_functions_via_code.ContainsKey(attr.code | DATA_CONVERTOR_SIGN))
                            throw new ArgumentException(
                                nameof(convertor), "this type of IDataConvertor is not registered.");
                        var serializer = registered_functions_via_code[attr.code | DATA_CONVERTOR_SIGN];
                        code = serializer.Code;
                        var buffer = serializer.Serialize(convertor);
                        if ((buffer?.Length ?? 0) != serializer.ParameterLength)
                            throw new Exception("invalid parameters' length.");
                        parameters = new List<byte>();
                        if (buffer != null) parameters.AddRange(buffer);
                        break;
                }

            // serialaize type and parameters
            Serialize(code, parameters?.ToArray());
        }
        public void Serialize(IRegularization regularization)
        {
            ushort code;
            byte[] parameters;
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
                        var attr = GetAttribute(regularization.GetType());
                        if (!registered_functions_via_code.ContainsKey(attr.code | REGULARIZATION_SIGN))
                            throw new ArgumentException(
                                nameof(regularization), "this type of IRegularization is not registered.");
                        var serializer = registered_functions_via_code[attr.code | REGULARIZATION_SIGN];
                        code = serializer.Code;
                        parameters = serializer.Serialize(regularization);
                        if ((parameters?.Length ?? 0) != serializer.ParameterLength)
                            throw new Exception("invalid parameters' length.");
                        break;
                }

            // serialaize type and parameters
            Serialize(code, parameters);
        }
        public void Serialize(IDataCombiner combiner)
        {
            ushort code;
            byte[] parameters;

            var attr = GetAttribute(combiner.GetType());
            if (!registered_functions_via_code.ContainsKey(attr.code | DATA_COMBINER_SIGN))
                throw new ArgumentException(
                    nameof(combiner), "this type of IDataCombiner is not registered.");
            var serializer = registered_functions_via_code[attr.code | DATA_COMBINER_SIGN];
            code = serializer.Code;
            parameters = serializer.Serialize(combiner);
            if ((parameters?.Length ?? 0) != serializer.ParameterLength)
                throw new Exception("invalid parameters' length.");

            // serialaize type and parameters
            Serialize(code, parameters);
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
            var code = RestorFunctionType();
            byte[] buffer;

            switch (code)
            {
                case 1: return new Errorest();
                case 2:
                    buffer = RestorParameter(4);
                    return new Deprecated.ErrorStack(BitConverter.ToInt32(buffer, 0));
                case 3:
                    buffer = RestorParameter(8);
                    return new Classification(BitConverter.ToDouble(buffer, 0));
                case 4:
                    buffer = RestorParameter(16);
                    return new Tagging(BitConverter.ToInt32(buffer, 0), BitConverter.ToInt32(buffer, 8));
                default:
                    if (!registered_functions_via_code.ContainsKey(code | ERROR_FUNCTION_SIGN))
                        throw new Exception(
                            $"this type of IErrorFunction ({code}) is not registered.");
                    return (IErrorFunction)Restore(registered_functions_via_code[code | ERROR_FUNCTION_SIGN]);
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
                case 2:
                    buffer = RestorParameter(16);
                    return new DataRangeDouble(
                        BitConverter.ToDouble(buffer, 0),
                        BitConverter.ToDouble(buffer, 8));

                default:
                    if (!registered_functions_via_code.ContainsKey(code | DATA_CONVERTOR_SIGN))
                        throw new Exception(
                            $"This type of IDataConvertor ({code}) is not registered.");
                    return (IDataConvertor)Restore(registered_functions_via_code[code | DATA_CONVERTOR_SIGN]);
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
                    if (!registered_functions_via_code.ContainsKey(code | REGULARIZATION_SIGN))
                        throw new Exception(
                            $"this type of IRegularization ({code}) is not registered.");
                    return (IRegularization)Restore(registered_functions_via_code[code | REGULARIZATION_SIGN]);
            }
        }
        public IDataCombiner RestoreIDataCombiner()
        {
            var code = RestorFunctionType();

            if (!registered_functions_via_code.ContainsKey(code | DATA_COMBINER_SIGN))
                throw new Exception(
                    $"this type of IDataCombiner ({code}) is not registered.");
            return (IDataCombiner)Restore(registered_functions_via_code[code | DATA_COMBINER_SIGN]);
        }
        private ISerializableFunction Restore(IFunctionSerializer serializer)
        {
            byte[] buffer;
            if (serializer.ParameterLength < 1) buffer = null;
            else buffer = RestorParameter(serializer.ParameterLength);

            return serializer.Restore(buffer);
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
                SoftMax _=> 5,
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
                5 => new SoftMax(),
                _ => throw new ArgumentException(
                    nameof(value), "this type of IConduction is not registered."),
            };
            all_conductions.Add(value, conduction);
            return conduction;
        }

    }
}