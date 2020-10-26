using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Opertat.Implement;

namespace Photon.NeuralNetwork.Opertat
{
    public class Brain
    {
        private const int lock_time_out = 1000;
        private readonly ReaderWriterLock locker = new ReaderWriterLock();

        private readonly Layer[] layers;
        private readonly IErrorFunction error_fnc;
        private readonly IDataConvertor in_cvrt, out_cvrt;
        private readonly IRegularization regularization;
        public double LearningFactor { get; set; } = 1;
        public double CertaintyFactor { get; set; } = 0;

        public Brain(NeuralNetworkImage image)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image),
                    "The nn-image is undefined.");

            NeuralNetworkImage.CheckImageError(image.layers, image.error_fnc);

            layers = image.layers;
            error_fnc = image.error_fnc;
            in_cvrt = image.input_convertor;
            out_cvrt = image.output_convertor;
            regularization = image.regularization;
        }

        public NeuralNetworkImage Image()
        {
            // it's for multi-thread using
            locker.AcquireReaderLock(lock_time_out);
            // generate image
            try
            {
                Layer[] clone = new Layer[layers.Length];
                for (int l = 0; l < clone.Length; l++)
                    clone[l] = layers[l].Clone();

                return new NeuralNetworkImage(
                    clone, error_fnc, in_cvrt, out_cvrt, regularization);
            }
            // release the lock
            finally { locker.ReleaseReaderLock(); }
        }

        public double[] Stimulate(double[] inputs)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            var signals = Vector<double>.Build.DenseOfArray(inputs);
            // standardized signals
            if (in_cvrt != null) signals = in_cvrt.Standardize(signals);

            // it's for multi-thread using
            locker.AcquireReaderLock(lock_time_out);
            try
            {
                var i = 0;
                for (; i < layers.Length; i++)
                {
                    // multiply inputs and weights plus bias
                    signals = layers[i].Synapse.Multiply(signals) + layers[i].Bias[i];
                    // apply sigmoind function on results
                    signals = layers[i].Conduction.Conduct(signals);
                }
            }
            finally { locker.ReleaseReaderLock(); }

            // normalaize result to return
            if (out_cvrt != null) signals = out_cvrt.Normalize(signals);
            // return result
            return signals.ToArray();
        }
        public NeuralNetworkFlash Test(double[] inputs)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            var signals = Vector<double>.Build.DenseOfArray(inputs);

            // standardized signals
            if (in_cvrt != null) signals = in_cvrt.Standardize(signals);
            // prepare neural network flash
            var flash = new NeuralNetworkFlash(layers.Length);

            // it's for multi-thread using
            locker.AcquireReaderLock(lock_time_out);
            try
            {
                // forward-propagation
                ForwardPropagation(flash, ref signals);
            }
            finally { locker.ReleaseReaderLock(); }

            // normalaize result to return
            if (out_cvrt != null) signals = out_cvrt.Normalize(signals);
            // set result
            flash.ResultSignals = signals.ToArray();

            return flash;
        }
        public void Reflect(NeuralNetworkFlash flash, double[] values)
        {
            if (flash == null)
                throw new ArgumentNullException(nameof(flash));
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            // pure data
            var delta = Vector<double>.Build.DenseOfArray(values);
            // standardized signals
            if (out_cvrt != null) delta = out_cvrt.Standardize(delta);
            // calculate error
            delta = error_fnc.ErrorCalculation(flash.InputSignals[^1], delta);

            // it's for multi-thread using
            locker.AcquireWriterLock(lock_time_out);
            try
            {
                // back-propagation
                BackPropagation(flash, delta);
            }
            finally { locker.ReleaseWriterLock(); }
        }
        public NeuralNetworkFlash Train(double[] inputs, double[] values)
        {
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            var signals = Vector<double>.Build.DenseOfArray(inputs);

            // standardized signals
            if (in_cvrt != null) signals = in_cvrt.Standardize(signals);
            // prepare neural network flash
            var flash = new NeuralNetworkFlash(layers.Length);

            // pure data
            var delta = Vector<double>.Build.DenseOfArray(values);
            // standardized signals
            if (out_cvrt != null) delta = out_cvrt.Standardize(delta);

            // it's for multi-thread using
            locker.AcquireWriterLock(lock_time_out);
            try
            {
                // dropout
                // use drop out *optional

                // forward-propagation
                ForwardPropagation(flash, ref signals);

                // calculate error
                delta = error_fnc.ErrorCalculation(flash.InputSignals[^1], delta);

                // back-propagation
                BackPropagation(flash, delta);
            }
            finally { locker.ReleaseWriterLock(); }

            // normalaize result to return
            if (out_cvrt != null) signals = out_cvrt.Normalize(signals);
            // set result
            flash.ResultSignals = signals.ToArray();

            return flash;
        }
        private void ForwardPropagation(NeuralNetworkFlash flash, ref Vector<double> signals)
        {
            var i = 0;
            for (; i < layers.Length; i++)
            {
                // store input for this layer
                flash.InputSignals[i] = signals;
                // multiply inputs and weights plus bias
                flash.SignalsSum[i] = layers[i].Synapse.Multiply(signals) + layers[i].Bias;
                // apply sigmoind function on results
                signals = layers[i].Conduction.Conduct(flash, i);
#if NaN && DEBUG
                NanTest(signals);
                //flash.SignalsExtra[0][i]
#endif
            }
            // store last output
            flash.InputSignals[i] = signals;
        }
        private void BackPropagation(NeuralNetworkFlash flash, Vector<double> delta)
        {
            var i = layers.Length;
            while (--i >= 0)
            {
                // calculate next delta
                delta = delta.PointwiseMultiply(
                    layers[i].Conduction.ConductDerivative(flash, i));

                // find out bias and weights differances
                var delta_bias = LearningFactor * delta;
                var delta_weight =
                    Matrix<double>.Build.DenseOfColumnVectors(delta_bias) *
                    Matrix<double>.Build.DenseOfRowVectors(flash.InputSignals[i]);

                // regularization
                if (CertaintyFactor > 0)
                    delta_weight -= regularization?.Regularize(layers[i].Synapse, CertaintyFactor);

                // prepare delta for next loop (previous layer)
                delta = layers[i].Synapse.Transpose().Multiply(delta);

                // apply bias and weights differances
                layers[i].Bias += delta_bias;
                layers[i].Synapse += delta_weight;
            }
        }

        private Vector<double> Error(NeuralNetworkFlash flash, double[] values)
        {
            if (flash == null)
                throw new ArgumentNullException(nameof(flash));
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            // calucate errors
            return error_fnc.ErrorCalculation(
                Vector<double>.Build.DenseOfArray(flash.ResultSignals),
                Vector<double>.Build.DenseOfArray(values));
        }
        public double[] Errors(NeuralNetworkFlash flash, double[] values)
        {
            var error = Error(flash, values);

            var result = error.AsArray();
            if (result == null) result = error.ToArray();

            return result;
        }
        public double ErrorTotal(NeuralNetworkFlash flash, double[] values)
        {
            return Error(flash, values).PointwiseAbs().Sum();
        }
        public double ErrorAverage(NeuralNetworkFlash flash, double[] values)
        {
            if (values.Length < 1) return 0;
            else return ErrorTotal(flash, values) / values.Length;
        }

        public double Accuracy(NeuralNetworkFlash flash, double[] values, out double error_sum)
        {
            if (flash == null)
                throw new ArgumentNullException(nameof(flash));
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (values.Length == 0)
            {
                error_sum = 0;
                return 0;
            }

            var criterion = Vector<double>.Build.DenseOfArray(values);
            // standardized signals
            if (out_cvrt != null) criterion = out_cvrt.Standardize(criterion);
            // calculate error
            criterion = error_fnc.ErrorCalculation(flash.InputSignals[^1], criterion);
            // calculate accuracy
            error_sum = criterion.PointwiseAbs().Sum();
            return 1 - error_sum / criterion.Count;
        }

#if NaN && DEBUG
        public static void NanTest(Layer layer)
        {
            NanTest(layer.Synapse);
            NanTest(layer.Bias);
        }
        public static void NanTest(Matrix<double> matrix)
        {
            for (int i = 0; i < matrix.RowCount; i++)
                for (int j = 0; j < matrix.ColumnCount; j++)
                    if (double.IsNaN(matrix[i, j]) || double.IsInfinity(matrix[i, j]))
                    {

                    }
        }
        public static void NanTest(Vector<double> vector)
        {
            for (int i = 0; i < vector.Count; i++)
                if (double.IsNaN(vector[i]) || double.IsInfinity(vector[i]))
                {

                }
        }
#endif
    }
}
