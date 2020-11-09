package Photon.NeuralNetwork.Opertat.Implement;

import org.ejml.data.DMatrixRMaj;

public interface IErrorFunction {

	public DMatrixRMaj ErrorCalculation(DMatrixRMaj output, DMatrixRMaj values);
    
}
