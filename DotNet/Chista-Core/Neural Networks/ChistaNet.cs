using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using Photon.NeuralNetwork.Chista.Implement;

namespace Photon.NeuralNetwork.Chista
{
    public class ChistaNet : IChistaNet, INeuralNetworkInformation
    {
        private const int lock_time_out = -1;
        private readonly ReaderWriterLock locker = new ReaderWriterLock();

        private readonly Layer[] layers;
        private readonly IErrorFunction error_fnc;
        private readonly IDataConvertor in_cvrt, out_cvrt;
        private readonly IRegularization regularization;

        public double LearningFactor { get; set; } = 0.01;
        public double CertaintyFactor { get; set; } = 0.001;
        public double DropoutFactor { get; set; } = 0.4;
        public int InputCount => layers[0].Synapse.ColumnCount;
        public int OutputCount => layers[^1].Synapse.RowCount;

        public ChistaNet(NeuralNetworkImage image)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "The nn-image is undefined.");

            NeuralNetworkImage.CheckImageError(image.layers, image.error_fnc);

            layers = new Layer[image.layers.Length];
            for (int l = 0; l < layers.Length; l++)
                layers[l] = image.layers[l].Clone();

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
        INeuralNetworkImage IChistaNet.Image()
        {
            return Image();
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
        public NeuralNetworkFlash Test(double[] inputs, double[] values = null)
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
                // don't use dropout

                // forward-propagation
                ForwardPropagation(flash, ref signals);
            }
            finally { locker.ReleaseReaderLock(); }

            // normalaize result to return
            if (out_cvrt != null) signals = out_cvrt.Normalize(signals);
            // set result
            flash.ResultSignals = signals.ToArray();
            // check error
            if (values != null) FillTotalError(flash, values);

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
            // calculate error and total error
            delta = error_fnc.ErrorCalculation(flash.InputSignals[^1], delta);
            // calculate total error of network result
            flash.SetErrors(delta);
            // set accuracy
            flash.Accuracy = error_fnc.Accuracy(flash);
            // check if is not any error then do not train the network
            if (flash.TotalError != 0) return;

            // it's for multi-thread using
            locker.AcquireWriterLock(lock_time_out);
            // double lr = LearningFactor;
            try
            {
                // if nodes are droped out then increase learning factor
                if (DropoutFactor > 0) LearningFactor /= DropoutFactor;
                // back-propagation
                BackPropagation(flash, delta);
            }
            finally
            {
                // the dropout is not used

                // release lock
                locker.ReleaseWriterLock();
            }
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
            double lr = LearningFactor;
            try
            {
                // dropout
                if (DropoutFactor > 0)
                {
                    var nodes = new HashSet<int>();
                    var last = layers[^1];
                    foreach (var layer in layers)
                        layer.Droupout(last == layer ? 0 : DropoutFactor, ref nodes);
                }

                // forward-propagation
                ForwardPropagation(flash, ref signals);

                // calculate error and total error of network result
                delta = error_fnc.ErrorCalculation(flash.InputSignals[^1], delta);
                // calculate total error of network result
                flash.SetErrors(delta);
                // set accuracy
                flash.Accuracy = error_fnc.Accuracy(flash);

                // check if is not any error then do not train the network
                if (flash.TotalError != 0)
                {
                    // if nodes are droped out then increase learning factor
                    if (DropoutFactor > 0) LearningFactor /= DropoutFactor;
                    // back-propagation
                    BackPropagation(flash, delta);
                }
            }
            finally
            {
                // release dropout
                if (DropoutFactor > 0)
                {
                    LearningFactor = lr;
                    var nodes = new HashSet<int>();
                    foreach (var layer in layers)
                        layer.DroupoutRelease(ref nodes);
                }

                // release lock
                locker.ReleaseWriterLock();
            }

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
#if NaN
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
                if (CertaintyFactor > 0 && regularization != null)
                    delta_weight -= regularization.Regularize(layers[i].Synapse, CertaintyFactor);

                // prepare delta for next loop (previous layer)
                delta = layers[i].Synapse.Transpose().Multiply(delta);

                // apply bias and weights differances
                layers[i].Bias += delta_bias;
                layers[i].Synapse += delta_weight;
            }
        }

        public void FillTotalError(NeuralNetworkFlash flash, double[] values)
        {
            if (flash == null)
                throw new ArgumentNullException(nameof(flash));
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            // pure data
            var delta = Vector<double>.Build.DenseOfArray(values);
            // standardized signals
            if (out_cvrt != null) delta = out_cvrt.Standardize(delta);
            // calculate error and total error of network result
            error_fnc.ErrorCalculation(flash.InputSignals[^1], delta);
            // calculate total error of network result
            flash.SetErrors(delta);
            // set accuracy
            flash.Accuracy = error_fnc.Accuracy(flash);
        }

#if NaN_CHEK
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
                        throw new Exception("NaN value");
        }
        public static void NanTest(Vector<double> vector)
        {
            for (int i = 0; i < vector.Count; i++)
                if (double.IsNaN(vector[i]) || double.IsInfinity(vector[i]))
                    throw new Exception("NaN value");
        }
#endif

        public override string ToString()
        {
            /* THIS CODE IS COPEIED FROM 'NeuralNetworkImage'.'ToString()' BECAUSE OF PERFORMANCE */
            var buffer = new StringBuilder()
                .Append("layers(").Append(layers.Length).Append("): ")
                .Append(layers[0].Synapse.ColumnCount);
            foreach (var l in layers)
                buffer.Append("x").Append(l.Synapse.RowCount);
            return buffer.ToString();
        }
        public string PrintInfo()
        {
            /* THIS CODE IS COPEIED FROM 'NeuralNetworkImage'.'PrintInfo()' BECAUSE OF PERFORMANCE */
            var buffer = new StringBuilder("[chista net]");

            if (in_cvrt != null)
                buffer.Append("\n")
                    .Append("input data convertor: ").Append(in_cvrt.ToString());

            if (layers != null)
            {
                buffer.Append("\n").Append("layers: ").Append(layers.Length);
                if (layers.Length > 0)
                {
                    buffer.Append("\n\t")
                        .Append("input: ").Append(layers[0].Synapse.ColumnCount).Append(" node(s)");
                    foreach (var l in layers)
                        buffer.Append("\n\t").Append("layer: ")
                            .Append(l.Synapse.RowCount).Append(" node(s)")
                            .Append(" func=").Append(l.Conduction.ToString());
                }
            }

            if (out_cvrt != null)
                buffer.Append("\n")
                    .Append("output data convertor: ").Append(out_cvrt.ToString());

            if (error_fnc != null)
                buffer.Append("\n")
                    .Append("error function: ").Append(error_fnc.ToString());

            if (regularization != null)
                buffer.Append("\n")
                    .Append("regularization: ").Append(regularization.ToString());

            return buffer.ToString();
        }

    }
}
