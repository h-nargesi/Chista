using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Photon.NeuralNetwork.Opertat.Implement
{
    public class Layer
    {
        private Matrix<double> dropout_synapse_backup;
        private HashSet<int> current_doped;

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

        public Layer(IConduction conduction)
        {
            Conduction = conduction;
        }

        public Layer Clone()
        {
            return new Layer(Conduction)
            {
                Synapse = Synapse.Clone(),
                Bias = Bias.Clone()
            };
        }

        public void Droupout(double percentage, ref HashSet<int> previous_doped)
        {
            if (dropout_synapse_backup != null)
                throw new Exception("The last droped node is not recovered.");

            // point index
            current_doped = new HashSet<int>();
			if (percentage > 0) {
				var random = new Random((int)DateTime.Now.Ticks);
				for (int i = 0; i < Synapse.ColumnCount; i++)
					if (Synapse.ColumnCount - current_doped.Count <= 1) break;
					else if (random.NextDouble() <= percentage) current_doped.Add(i);
			}

            var new_matrix = new double[
                Synapse.RowCount - previous_doped.Count,
                Synapse.ColumnCount - current_doped.Count];
            for (int ro = 0, rn = 0; ro < Synapse.RowCount; ro++)
                if (previous_doped.Contains(ro)) continue;
                else
                {
                    for (int co = 0, cn = 0; co < Synapse.ColumnCount; co++)
                        if (current_doped.Contains(co)) continue;
                        else
                        {
                            new_matrix[rn, cn] = Synapse[ro, co];
                            cn++;
                        }
                    rn++;
                }

            dropout_synapse_backup = Synapse;
            Synapse = Matrix<double>.Build.DenseOfArray(new_matrix);
            previous_doped = current_doped;
        }
        public void DroupoutRelease(ref HashSet<int> previous_doped)
        {
            if (dropout_synapse_backup == null) return;

			
            for (int ro = 0, rn = 0; ro < dropout_synapse_backup.RowCount; ro++)
                if (previous_doped.Contains(ro)) continue;
                else
                {
                    for (int co = 0, cn = 0; co < dropout_synapse_backup.ColumnCount; co++)
                        if (current_doped.Contains(co)) continue;
                        else
                        {
                            dropout_synapse_backup[rn, cn] = Synapse[ro, co];
                            cn++;
                        }
                    rn++;
                }

			Synapse = dropout_synapse_backup;
			previous_doped = current_doped;
			dropout_synapse_backup = null;
			current_doped = null;
        }
    }
}