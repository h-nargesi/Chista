package Photon.NeuralNetwork.Opertat.Implement;

import org.ejml.data.DMatrixRMaj;

public interface IRegularization {

	public DMatrixRMaj Regularize(DMatrixRMaj synapse, double certainty);
    
}
