package com.cefetmg.perceptron.utils.model;

public class DataSetSplitResult {

    public Double[][] trainingInput;
    public Double[][] trainingOutput;
    public Double[][] testingInput;
    public Double[][] testingOutput;

    public DataSetSplitResult(Double[][] trainingInput, Double[][] trainingOutput, Double[][] testingInput, Double[][] testingOutput) {
        this.trainingInput = trainingInput;
        this.trainingOutput = trainingOutput;
        this.testingInput = testingInput;
        this.testingOutput = testingOutput;
    }
}
