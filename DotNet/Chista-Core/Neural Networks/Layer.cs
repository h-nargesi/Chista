using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Chista.Implement
{
    public class Layer
    {
        private Matrix<double> dropout_synapse_backup;
        private Vector<double> dropout_bias_backup;
        private HashSet<int> current_droped;

        public Matrix<double> Synapse { get; set; }
        public Vector<double> Bias { get; set; }
        public IConduction Conduction { get; }
        public Matrix<double> SafeSynapse
        {
            get
            {
                if (dropout_synapse_backup == null) return Synapse;
                else return dropout_synapse_backup;
            }
        }
        public Vector<double> SafeBias
        {
            get
            {
                if (dropout_bias_backup == null) return Bias;
                else return dropout_bias_backup;
            }
        }

        public Layer(IConduction conduction)
        {
            Conduction = conduction;
        }

        public Layer Clone()
        {
            return new Layer(Conduction)
            {
                Synapse = SafeSynapse.Clone(),
                Bias = SafeBias.Clone()
            };
        }

        public void Droupout(double percentage, ref HashSet<int> previous_droped)
        {
            if (dropout_synapse_backup != null)
                throw new Exception("The last droped node is not recovered.");

            // point index
            current_droped = new HashSet<int>();
			if (percentage > 0) {
				var random = new Random((int)DateTime.Now.Ticks);
				for (int i = 0; i < Synapse.RowCount; i++)
					if (Synapse.RowCount - current_droped.Count <= 1) break;
					else if (random.NextDouble() <= percentage) current_droped.Add(i);
			}

            var new_synaps = new double[
                Synapse.RowCount - current_droped.Count,
                Synapse.ColumnCount - previous_droped.Count];
            var new_bias = new double[Bias.Count - current_droped.Count];

            for (int ro = 0, rn = 0; ro < Synapse.RowCount; ro++)
                if (current_droped.Contains(ro)) continue;
                else
                {
                    new_bias[rn] = Bias[ro];

                    for (int co = 0, cn = 0; co < Synapse.ColumnCount; co++)
                        if (previous_droped.Contains(co)) continue;
                        else
                        {
                            new_synaps[rn, cn] = Synapse[ro, co];
                            cn++;
                        }
                    rn++;
                }

            // backup current state
            dropout_synapse_backup = Synapse;
            dropout_bias_backup = Bias;
            // change to new state
            Synapse = Matrix<double>.Build.DenseOfArray(new_synaps);
            Bias = Vector<double>.Build.DenseOfArray(new_bias);

            previous_droped = current_droped;
        }
        public void DroupoutRelease(ref HashSet<int> previous_droped)
        {
            if (dropout_synapse_backup == null) return;

            for (int ro = 0, rn = 0; ro < dropout_synapse_backup.RowCount; ro++)
                if (current_droped.Contains(ro)) continue;
                else
                {
                    dropout_bias_backup[ro] = Bias[rn];

                    for (int co = 0, cn = 0; co < dropout_synapse_backup.ColumnCount; co++)
                        if (previous_droped.Contains(co)) continue;
                        else
                        {
                            dropout_synapse_backup[ro, co] = Synapse[rn, cn];
                            cn++;
                        }
                    rn++;
                }

            // restore to last state
            Synapse = dropout_synapse_backup;
            Bias = dropout_bias_backup;
            // empty memory
			dropout_synapse_backup = null;
            dropout_bias_backup = null;
            // return droped
            previous_droped = current_droped;
            current_droped = null;
        }

        public override string ToString()
        {
            return $"Layer[{Synapse.ColumnCount}x{Synapse.RowCount}].{Conduction}";
        }
    }
}