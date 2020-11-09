package Photon.NeuralNetwork.Opertat.Implement;

import org.ejml.data.DMatrixRMaj;

import Photon.NeuralNetwork.Opertat.NeuralNetworkFlash;

public interface IConduction {
	
	public int ExtraCount();
	public DMatrixRMaj Conduct(DMatrixRMaj signal);
	public DMatrixRMaj Conduct(NeuralNetworkFlash flash, int layer);
	public DMatrixRMaj ConductDerivative(NeuralNetworkFlash flash, int layer);
	
}
