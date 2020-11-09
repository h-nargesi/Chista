package Photon.NeuralNetwork.Opertat.Implement;

import java.util.HashSet;

import org.ejml.data.DMatrixRMaj;

public class Layer {

	private DMatrixRMaj dropout_synapse_backup, dropout_bias_backup;
	private HashSet<Integer> current_droped;

	public DMatrixRMaj synapse, bias;
	public final IConduction conduction;

	public Layer(IConduction conduction) {
		this.conduction = conduction;
	}

	public Layer Clone() {
		Layer layer = new Layer(conduction);
		layer.synapse = synapse.copy();
		layer.bias = bias.copy();
		return layer;
	}
}
