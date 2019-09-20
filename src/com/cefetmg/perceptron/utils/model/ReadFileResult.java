package com.cefetmg.perceptron.utils.model;

public class ReadFileResult {

    public Double[][] inputs;
    public Double[][] targets;
    public Double[][] testingInput;
    public Double[][] testingOutput;

    public ReadFileResult(Double[][] inputs, Double[][] targets, Double[][] testingInput, Double[][] testingOutput) {
        this.inputs = inputs;
        this.targets = targets;
        this.testingInput = testingInput;
        this.testingOutput = testingOutput;
    }
}
