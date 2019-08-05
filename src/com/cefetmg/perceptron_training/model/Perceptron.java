package com.cefetmg.perceptron_training.model;

import com.cefetmg.perceptron_training.utils.MathFunctions;

public class Perceptron {

    private Integer input;
    private Integer output;
    private Double[][] weights;

    public Perceptron(Integer input, Integer output) {
        this.input = input;
        this.output = output;
        this.weights = new Double[this.input + 1][this.output];
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = weightsInitialize(this.output);
        }
    }

    private Double[] weightsInitialize(int size) {
        Double[] list = new Double[size];
        for (int i = 0; i < size; i++) {
            list[i] = Math.random() * 0.2 - Math.random() * 0.2; // range: [-0.2, 0.2]
        }
        return list;
    }

    public Double[] training(Double[] sources, Double[] targets, Double m) {
        if (sources == null || targets == null) {
            return null;
        }

        Double[] sourcesWithBias = new Double[sources.length + 1];
        sourcesWithBias[0] = 1d; // bias
        for (int i = 1; i < sourcesWithBias.length; i++) {
            sourcesWithBias[i] = sources[i - 1];
        }

        Double[] output = new Double[targets.length];

        for (int j = 0; j < targets.length; j++) {
            Double u = 0d;
            for (int i = 0; i < sourcesWithBias.length; i++) {
                u += sourcesWithBias[i] * this.weights[i][j];
            }
            output[j] = MathFunctions.sigmoidal(u);
        }

        Double[][] deltas = new Double[this.input + 1][this.output];

        for (int j = 0; j < targets.length; j++) {
            for (int i = 0; i < sourcesWithBias.length; i++) {
                deltas[i][j] = m * (targets[j] - output[j]) * sourcesWithBias[i];
                this.weights[i][j] += deltas[i][j];
            }
        }

        return output;
    }
}
