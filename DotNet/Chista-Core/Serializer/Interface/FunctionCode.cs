using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Serializer
{
	[AttributeUsage(AttributeTargets.Class)]
	public class FunctionCode : Attribute
	{
		public readonly ushort code;
		public readonly int parameter_length;

		public FunctionCode(ushort code, int parameter_length)
		{
			this.code = code;
			this.parameter_length = parameter_length;
		}
	}
}
