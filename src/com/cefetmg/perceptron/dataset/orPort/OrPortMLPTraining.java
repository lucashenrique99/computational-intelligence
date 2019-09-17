package com.cefetmg.perceptron.dataset.orPort;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;
import com.cefetmg.perceptron.model.Perceptron;

import java.util.Arrays;

public class OrPortMLPTraining {

    public static void main(String[] args) {

        Double[][] source = new Double[][]{{0d, 0d}, {0d, 1d}, {1d, 0d}, {1d, 1d}};
        Double[][] target = new Double[][]{{0d}, {1d}, {1d}, {1d}};

        MultiLayerPerceptron perceptron = new MultiLayerPerceptron(2, 2,  1);

        for (int epoch = 0; epoch < 5000; epoch++) {
            double epochError = 0;
            int epochErrorClassifier = 0;
            Double threShold = 0.5d;
            for (int i = 0; i < source.length; i++) {
                Double[] input = source[i];
                Double[] output = target[i];
                Double[] result = perceptron.training(input, output, 0.3);
                double sampleError = 0;
                int sampleErrorClassifier = 0;
                for (int j = 0; j < result.length; j++) {
                    Double e = Math.abs(result[j] - output[j]);;
                    sampleError += e;
                    Double outValue = (result[j] < threShold) ? 0 : 1d; // truncate value
                    sampleErrorClassifier += Math.abs(output[j] - outValue);
                }
                epochError += sampleError;
                epochErrorClassifier += sampleErrorClassifier;
            }

            System.out.println("Epoch " + epoch + ":  error: " + epochError + " \t  error classifier: " + epochErrorClassifier);
        }

        Double[] sourceTest = {1d, 1d};
        Double[] out = perceptron.getTargetBySource(sourceTest);
        System.out.println(Arrays.toString(out));
    }

}
