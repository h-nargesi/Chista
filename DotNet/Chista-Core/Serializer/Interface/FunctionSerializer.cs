using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Reflection;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista.Serializer
{
    public interface IFunctionSerializer
    {
        string Name { get; }
        ushort Code { get; }
        int Ucode { get; }
        int ParameterLength { get; }
        Type FunctionType { get; }

        byte[] Serialize(ISerializableFunction func);
        ISerializableFunction Restore(byte[] parameters);
    }

    public abstract class FunctionSerializer<T> : IFunctionSerializer where T : ISerializableFunction
    {
        public FunctionSerializer()
        {
            FunctionType = typeof(T);

            var attr = FunctionSerializerCore.GetAttribute(FunctionType);

            if (attr.code < 0x8000)
                throw new ArgumentException(FunctionType.Name,
                    $"Invalid code({attr.code}). The code must be greater than 32768 (0x8000).");

            Ucode = Code = attr.code;
            if (typeof(IErrorFunction).IsAssignableFrom(FunctionType))
                Ucode |= FunctionSerializerCore.ERROR_FUNCTION_SIGN;
            if (typeof(IDataConvertor).IsAssignableFrom(FunctionType))
                Ucode |= FunctionSerializerCore.DATA_CONVERTOR_SIGN;
            if (typeof(IRegularization).IsAssignableFrom(FunctionType))
                Ucode |= FunctionSerializerCore.REGULARIZATION_SIGN;

            if (Code == Ucode)
                throw new ArgumentException(FunctionType.Name,
                    $"The type if serializer ({FunctionType.Name}) is invalid.");

            if (attr.parameter_length < 0)
                throw new ArgumentException(FunctionType.Name,
                    $"Invalid parameter's length for type({FunctionType.Name}).");
            ParameterLength = attr.parameter_length;
        }

        public abstract string Name { get; }

        public abstract byte[] Serialize(T func);
        public byte[] Serialize(ISerializableFunction func)
        {
            if (func is T t_func) return Serialize(t_func);
            else throw new ArgumentException(nameof(func), "Invalid function type");
        }

        public abstract T Restore(byte[] parameters);
        ISerializableFunction IFunctionSerializer.Restore(byte[] parameters)
        {
            return Restore(parameters);
        }

        public ushort Code { get; }
        public int Ucode { get; }
        public int ParameterLength { get; }
        public Type FunctionType { get; }

        public void Register()
        {
            FunctionSerializerCore.RegisterFunction(this);
        }
    }
}
