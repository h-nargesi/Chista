package Photon.NeuralNetwork.Opertat;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import Photon.NeuralNetwork.Opertat.Implement.*;

public class Brain {

	private final static int lock_time_out = 1000;
	private final ReadWriteLock locker = new ReentrantReadWriteLock();

	private final Layer[] layers;
	private final IErrorFunction error_fnc;
	private final IDataConvertor in_cvrt, out_cvrt;
	private final IRegularization regularization;

	public double learning_factor = 0.01;
	public double certainty_factor = 0.001;
	public double dropout_factor = 0.4;
	
	public Brain() {
		layers = null;
		error_fnc = null;
		in_cvrt = null;
		out_cvrt = null;
		regularization = null;
	}

	public double LearningFactor() {
		return learning_factor;
	}

	public Brain LearningFactor(double learning_factor) {
		this.learning_factor = learning_factor;
		return this;
	}

	public double CertaintyFactor() {
		return certainty_factor;
	}

	public Brain CertaintyFactor(double certainty_factor) {
		this.certainty_factor = certainty_factor;
		return this;
	}

	public double DropoutFactor() {
		return dropout_factor;
	}

	public Brain DropoutFactor(double dropout_factor) {
		this.dropout_factor = dropout_factor;
		return this;
	}
}
