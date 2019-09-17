package com.cefetmg.perceptron.model;

import com.cefetmg.perceptron.utils.MathFunctions;

public class MultiLayerPerceptron {

    private Integer input;
    private Integer intermediateNeurons;
    private Integer output;
    private Double[][] intermediateWeights;
    private Double[][] outputWeights;

    public MultiLayerPerceptron(Integer input, Integer intermediateNeurons, Integer output) {
        this.input = input;
        this.intermediateNeurons = intermediateNeurons;
        this.output = output;
        this.intermediateWeights = new Double[this.input + 1][this.intermediateNeurons];
        for (int i = 0; i < this.intermediateWeights.length; i++) {
            this.intermediateWeights[i] = weightsInitialize(this.intermediateNeurons);
        }

        this.outputWeights = new Double[this.intermediateNeurons + 1][this.output];
        for (int i = 0; i < this.outputWeights.length; i++) {
            this.outputWeights[i] = weightsInitialize(this.output);
        }
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

        Double[] intermediateOutput = new Double[this.intermediateNeurons + 1];
        intermediateOutput[this.intermediateNeurons] = 1d;

        for (int h = 0; h < this.intermediateNeurons; h++) {
            Double u = 0d;
            for (int i = 0; i < sourcesWithBias.length; i++) {
                u += sourcesWithBias[i] * this.intermediateWeights[i][h];
            }
            intermediateOutput[h] = MathFunctions.sigmoidal(u);
        }

        Double[] output = new Double[targets.length];

        for (int j = 0; j < targets.length; j++) {
            Double u = 0d;
            for (int i = 0; i < sourcesWithBias.length; i++) {
                u += intermediateOutput[i] * this.outputWeights[i][j];
            }
            output[j] = MathFunctions.sigmoidal(u);
        }

        Double[] outputDeltas = new Double[targets.length];
        for (int j = 0; j < targets.length; j++) {
            outputDeltas[j] = output[j] * (1 - output[j]) * (targets[j] - output[j]);
        }

        Double[] intermediateDeltas = new Double[this.intermediateNeurons];
        for (int h = 0; h < this.intermediateNeurons; h++) {
            double sum = 0;
            for (int j = 0; j < this.output; j++) {
                sum += outputDeltas[j] * this.outputWeights[h][j];
            }
            intermediateDeltas[h] = intermediateOutput[h] * (1 - intermediateOutput[h]) * sum;
        }

        for (int h = 0; h < this.intermediateNeurons; h++) {
            for (int i = 0; i < sourcesWithBias.length; i++) {
                this.intermediateWeights[i][h] += m * intermediateDeltas[h] * sourcesWithBias[i];
            }
        }

        for (int j = 0; j < output.length; j++) {
            for (int h = 0; h < intermediateOutput.length; h++) {
                this.outputWeights[h][j] += m * outputDeltas[j] * intermediateOutput[h];
            }
        }

        return output;
    }


    public Double[] getTargetBySource(Double[] source) {
        if (source == null) {
            return null;
        }

        Double[] sourcesWithBias = new Double[source.length + 1];
        sourcesWithBias[0] = 1d; // bias
        for (int i = 1; i < sourcesWithBias.length; i++) {
            sourcesWithBias[i] = source[i - 1];
        }

        Double[] intermediateOutput = new Double[this.intermediateNeurons + 1];
        intermediateOutput[this.intermediateNeurons] = 1d;

        for (int h = 0; h < this.intermediateNeurons; h++) {
            Double u = 0d;
            for (int i = 0; i < sourcesWithBias.length; i++) {
                u += sourcesWithBias[i] * this.intermediateWeights[i][h];
            }
            intermediateOutput[h] = MathFunctions.sigmoidal(u);
        }

        Double[] output = new Double[this.output];

        for (int j = 0; j < this.output; j++) {
            Double u = 0d;
            for (int i = 0; i < sourcesWithBias.length; i++) {
                u += intermediateOutput[i] * this.outputWeights[i][j];
            }
            output[j] = MathFunctions.sigmoidal(u);
        }

        return output;
    }

    private Double[] weightsInitialize(int size) {
        Double[] list = new Double[size];
        for (int i = 0; i < size; i++) {
            list[i] = Math.random() * 0.4 - 0.2; // range: [-0.2, 0.2]
        }
        return list;
    }

}
