package Photon.NeuralNetwork.Opertat;

import java.util.ArrayList;
import java.util.List;

import org.ejml.data.DMatrixRMaj;

public class NeuralNetworkFlash {

	final DMatrixRMaj[] signals_sum;
	final DMatrixRMaj[] input_signals;
	private final List<DMatrixRMaj[]> signals_extra;
	double[] result_signals;

    public NeuralNetworkFlash(int size)
    {
    	signals_sum = new DMatrixRMaj[size];
    	input_signals = new DMatrixRMaj[size + 1];
    	signals_extra = new ArrayList<DMatrixRMaj[]>();
    }

	public double[] ResultSignals() {
		return result_signals;
	}

}
