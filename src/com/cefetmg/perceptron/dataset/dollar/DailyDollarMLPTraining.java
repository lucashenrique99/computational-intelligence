package com.cefetmg.perceptron.dataset.dollar;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;
import com.cefetmg.perceptron.utils.FileUtils;
import com.cefetmg.perceptron.utils.Optimizations;
import com.cefetmg.perceptron.utils.XYLineChart;
import com.cefetmg.perceptron.utils.model.ReadFileResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;


public class DailyDollarMLPTraining {

    public static void main(String[] args) {
        execute();
    }

    public static void execute() {

        ReadFileResult readFileResult = readAllDataSetFile();
        Double[][] source = readFileResult.inputs;
        Double[][] target = readFileResult.targets;

        MultiLayerPerceptron dollarPerceptron = new MultiLayerPerceptron(16, 5, 1);

        final double learningCoefficient = 0.001d;

        double minEpochClassifier = Double.MAX_VALUE;
        double maxEpochClassifier = 0;

        double minEpochError = Double.MAX_VALUE;
        double maxEpochError = 0;

        int numEpochs = 40000;
        Double[][][] values = new Double[2][][];
        values[0] = new Double[numEpochs][2]; // training
        values[1] = new Double[numEpochs][2]; // testing
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double epochError = 0;
            int epochErrorClassifier = 0;
            int sampleTrainingIndex = 0;
            for (; sampleTrainingIndex < source.length; sampleTrainingIndex++) {

                Double[] input = source[sampleTrainingIndex];
                Double[] output = target[sampleTrainingIndex]; // [ 1 0 0] | [0 0 1] | [0 1 0]

                Double[] dollarResult = dollarPerceptron.training(input, new Double[]{output[0]}, learningCoefficient);
                Optimizations.thresholdChangeRange(dollarResult);

                double sampleError = 0;
                int sampleErrorClassifier = 0;

                // up perceptron
                Double dollarError = Math.abs(dollarResult[0] - output[0]); // error
                sampleError += dollarError;
                Double dollarOutValue = Optimizations.thresholdErrorTruncate(dollarResult[0]); // classifier error
                sampleErrorClassifier += Math.abs(output[0] - dollarOutValue);

                // group normal, suspect and pathological errors
                epochError += sampleError;
                epochErrorClassifier += Math.min(sampleErrorClassifier, 1);
            }

            minEpochClassifier = Math.min(minEpochClassifier, epochErrorClassifier);
            maxEpochClassifier = Math.max(maxEpochClassifier, epochErrorClassifier);
            minEpochError = Math.min(minEpochError, epochError);
            maxEpochError = Math.max(maxEpochError, epochError);

            System.out.print("\nEPOCH " + epoch + ": ");
            System.out.print("\tError Training: " + epochError);
            System.out.print("\tError Training classifier: " + epochErrorClassifier);
            values[0][epoch] = new Double[]{(double) epoch, ((double) epochErrorClassifier / source.length)};

            int sampleTestingIndex = 0;
            double epochTestingError = 0;
            int epochErrorTestingClassifier = 0;
            for (; sampleTestingIndex < readFileResult.testingInput.length; sampleTestingIndex++) {
                Double[] out1 = dollarPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);

                // up perceptron
                Double dollarError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]); // error
                epochTestingError += dollarError;

                Double dollarOutValue = Optimizations.thresholdErrorTruncate(out1[0]); // classifier error
                epochErrorTestingClassifier += Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - dollarOutValue);

            }

            System.out.print("\tError Testing: " + epochTestingError);
            System.out.print("\tError Testing classifier: " + epochErrorTestingClassifier);
            values[1][epoch] = new Double[]{(double) epoch, ((double) epochErrorTestingClassifier / readFileResult.testingInput.length)};
        }

        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/dollar/savedObjects/", "dollarPerceptron", dollarPerceptron);

        XYLineChart.showChart("Dollar Training", new String[]{"Training", "Testing"}, values);
        System.out.println();
        System.out.println("Classifier -> \t min " + minEpochClassifier + " \t max " + maxEpochClassifier);
        System.out.println("Error -> \t min " + minEpochError + " \t max " + maxEpochError);

    }

    /*
        output codifications ->
            U -> 1 0
            D -> 0 1
     */
    public static ReadFileResult readAllDataSetFile() {
        try {
            InputStreamReader inputStreamReaderInputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/dollar/files/entradas_dolar_horas_mod.txt")));
            BufferedReader bufferedReaderInputs = new BufferedReader(inputStreamReaderInputs);

            int trainingDataSetSize = 18;
            int testDataSetSize = 9;

            String line;
            Double[][] inputs = new Double[trainingDataSetSize][];
            Double[][] outputs = new Double[trainingDataSetSize][];
            Double[][] testingInput = new Double[testDataSetSize][];
            Double[][] testingOutput = new Double[testDataSetSize][];
            int index = 0;
            while ((line = bufferedReaderInputs.readLine()) != null) {

                String[] inputVector = line.split("\\s+");
                List<Integer> ignoredColumns = Arrays.asList(0, 10, 11);
                Double[] input = new Double[inputVector.length - ignoredColumns.size() - 1];
                Double[] output = new Double[1];

                int inputIndex = 0;
                for (int i = 0; i < inputVector.length; i++) {
                    if (!ignoredColumns.contains(i)) {
                        if (i == inputVector.length - 1) {
                            output[0] = Double.parseDouble(inputVector[i]);
                        } else {
                            input[inputIndex] = Double.parseDouble(inputVector[i]);
                            inputIndex++;
                        }
                    }
                }


                if (index < trainingDataSetSize) {
                    inputs[index] = input;
                    outputs[index] = output;
                } else {
                    testingInput[index - trainingDataSetSize] = input;
                    testingOutput[index - trainingDataSetSize] = output;
                }

                index++;
            }

            return new ReadFileResult(inputs, outputs, testingInput, testingOutput);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}
