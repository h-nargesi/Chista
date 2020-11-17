package Photon.NeuralNetwork.Opertat.Implement;

import org.ejml.data.DMatrixRMaj;

public interface IDataConvertor {

	public DMatrixRMaj Standardize(DMatrixRMaj values);
	public DMatrixRMaj Normalize(DMatrixRMaj values);
}
