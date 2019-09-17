package com.cefetmg.perceptron.dataset.breastCancer;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;
import com.cefetmg.perceptron.model.Perceptron;
import com.cefetmg.perceptron.utils.XYLineChart;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class BreastCancerTraining {

    public static void main(String[] args) {

        ReadFileResult readFileResult = readInputFile(true);
        Double[][] source = readFileResult.inputs;
        Double[][] target = readFileResult.targets;

        Perceptron benignPerceptron = new Perceptron(9, 1);
        Perceptron malignantPerceptron = new Perceptron(9, 1);

        final Double learningCoefficient = 0.05d;
        final Double threshold = 0.5d;

        double minEpochClassifier = Double.MAX_VALUE;
        double maxEpochClassifier = 0;

        double minEpochError = Double.MAX_VALUE;
        double maxEpochError = 0;

        int numEpochs = 5000;
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

                Double[] benignResult = benignPerceptron.training(input, new Double[]{output[0]}, learningCoefficient);
                Double[] malignResult = malignantPerceptron.training(input, new Double[]{output[1]}, learningCoefficient);

                double sampleError = 0;
                int sampleErrorClassifier = 0;

                // benign perceptron
                Double benignError = Math.abs(benignResult[0] - output[0]); // error
                sampleError += benignError;
                Double benignOutValue = (benignResult[0] < threshold) ? 0 : 1d; // classifier error
                sampleErrorClassifier += Math.abs(output[0] - benignOutValue);

                // malign perceptron
                Double malignError = Math.abs(malignResult[0] - output[1]);// error
                sampleError += malignError;
                Double malignOutValue = (malignResult[0] < threshold) ? 0 : 1d; // classifier error
                sampleErrorClassifier += Math.abs(output[1] - malignOutValue);

                // group left, balance and right errors
                epochError += sampleError;
                epochErrorClassifier += Math.min(sampleErrorClassifier, 1);
            }

            minEpochClassifier = Math.min(minEpochClassifier, epochErrorClassifier);
            maxEpochClassifier = Math.max(maxEpochClassifier, epochErrorClassifier);
            minEpochError = Math.min(minEpochError, epochError);
            maxEpochError = Math.max(maxEpochError, epochError);

//            System.out.println("================ EPOCH " + epoch + " ====================");
//            System.out.println("Error Training: " + epochError);
//            System.out.println("Error Training classifier: " + epochErrorClassifier + "\t Percent: " + ((double) epochErrorClassifier / sampleTrainingIndex));

            System.out.print("\nEPOCH " + epoch + ": ");
            System.out.print("\tError Training: " + epochError);
            System.out.print("\tError Training classifier: " + epochErrorClassifier);
            values[0][epoch] = new Double[]{(double) epoch, ((double) epochErrorClassifier / sampleTrainingIndex)};

            int sampleTestingIndex = 0;
            double epochTestingError = 0;
            int epochErrorTestingClassifier = 0;
            for (; sampleTestingIndex < readFileResult.testingInput.length; sampleTestingIndex++) {
//                System.out.println("================ TEST " + sampleTestingIndex + " ====================");
                double sampleTestingError = 0;
                Double[] out1 = benignPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out2 = malignantPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);

                if (Math.round(out1[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][0]) ||
                        Math.round(out2[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][1])) {
                    epochErrorTestingClassifier++;
                }

                // left perceptron
                Double benignError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]); // error
                sampleTestingError += benignError;

                // balance perceptron
                Double malignError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][1] - out2[0]);// error
                sampleTestingError += malignError;

                epochTestingError += sampleTestingError;
            }
//            System.out.println("Error Testing: " + epochTestingError);
//            System.out.println("Error Testing classifier: " + epochErrorTestingClassifier + "\t Percent: " + ((double) epochErrorTestingClassifier / sampleTestingIndex));
            System.out.print("\tError Testing: " + epochTestingError);
            System.out.print("\tError Testing classifier: " + epochErrorTestingClassifier);
            values[1][epoch] = new Double[]{(double) epoch, ((double) epochErrorTestingClassifier / sampleTestingIndex)};
        }

        XYLineChart.showChart("Breast Cancer Training", new String[]{"Training", "Testing"}, values);
        System.out.println();
        System.out.println("Classifier -> \t min " + minEpochClassifier + " \t max " + maxEpochClassifier);
        System.out.println("Error -> \t min " + minEpochError + " \t max " + maxEpochError);

//        for (int i = 0; i < readFileResult.testingInput.length; i++) {
//            System.out.println("================ TEST " + i + " ====================");
//            System.out.println(Arrays.toString(readFileResult.testingOutput[i]));
//            testResults(readFileResult.testingInput[i], benignPerceptron, malignantPerceptron, rightPerceptron);
//        }
    }

    private static void testResults(Double[] source, Perceptron p1, Perceptron p2, Perceptron p3) {
        Double[] out1 = p1.getTargetBySource(source);
        Double[] out2 = p2.getTargetBySource(source);
        Double[] out3 = p3.getTargetBySource(source);
        System.out.println("[ " + Math.round(out1[0]) + ", " + Math.round(out2[0]) + ", " + Math.round(out3[0]) + " ] ");
//        System.out.println(Arrays.toString(out1) + Arrays.toString(out2) + Arrays.toString(out3));
    }

    /*
        output codifications ->
            L -> 1 0 0
            B -> 0 1 0
            R -> 0 0 1
     */
    public static ReadFileResult readInputFile(boolean slitDataSet) {
        try {
            InputStreamReader ir = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/breastCancer/files/breast-cancer-wisconsin.data")));
            BufferedReader in = new BufferedReader(ir);

            int datasetSize = 699;

            String line;
            Double[][] inputs = new Double[datasetSize][];
            Double[][] outputs = new Double[datasetSize][];
            int index = 0;
            while ((line = in.readLine()) != null) {

                String[] vector = line.split(",");
                for (int i = 0; i < vector.length; i++) {
                    vector[i] = vector[i].equalsIgnoreCase("?") ? "0" : vector[i];
                }

                String className = vector[10];

                Double[] output = new Double[]{
                        (className.equalsIgnoreCase("2")) ? 1d : 0,
                        (className.equalsIgnoreCase("4")) ? 1d : 0
                };

                Double[] input = new Double[]{
                        Double.parseDouble(vector[1]),
                        Double.parseDouble(vector[2]),
                        Double.parseDouble(vector[3]),
                        Double.parseDouble(vector[4]),
                        Double.parseDouble(vector[5]),
                        Double.parseDouble(vector[6]),
                        Double.parseDouble(vector[7]),
                        Double.parseDouble(vector[8]),
                        Double.parseDouble(vector[9])
                };

                inputs[index] = input;
                outputs[index] = output;

                index++;
            }

            Double[][] testingInput = new Double[inputs.length][inputs[0].length];
            Double[][] testingOutput = new Double[outputs.length][outputs[0].length];

            // if the data set will be divide, this lines below should be used
            if (slitDataSet) {
                DataSetSplitResult dataSetSplitResult = dataSetSplit(inputs, outputs);
                inputs = dataSetSplitResult.trainingInput;
                outputs = dataSetSplitResult.trainingOutput;
                testingInput = dataSetSplitResult.testingInput;
                testingOutput = dataSetSplitResult.testingOutput;
            }

            return new ReadFileResult(inputs, outputs, testingInput, testingOutput);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static DataSetSplitResult dataSetSplit(Double[][] inputs, Double[][] outputs) {
        final double trainingPercent = 0.75d;

        ArrayList<Double[]> benignInput = new ArrayList<>();
        List<Double[]> benignOutput = new ArrayList<>();
        List<Double[]> malignInput = new ArrayList<>();
        List<Double[]> malignOutput = new ArrayList<>();
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i][0] == 1) {
                benignInput.add(inputs[i]);
                benignOutput.add(outputs[i]);
            } else if (outputs[i][1] == 1) {
                malignInput.add(inputs[i]);
                malignOutput.add(outputs[i]);
            }else {
                System.out.println("ERROR");
            }
        }

        Double[][] inputsTemp = new Double[benignInput.size()][];
        Double[][] outputTemp = new Double[benignInput.size()][];
        for (int i = 0; i < benignInput.size(); i++) {
            inputsTemp[i] = benignInput.get(i);
            outputTemp[i] = benignOutput.get(i);
        }
        DataSetSplitResult benignResult = dataSetSplit(inputsTemp, outputTemp, trainingPercent);

        inputsTemp = new Double[malignInput.size()][];
        outputTemp = new Double[malignOutput.size()][];
        for (int i = 0; i < malignInput.size(); i++) {
            inputsTemp[i] = malignInput.get(i);
            outputTemp[i] = malignOutput.get(i);
        }
        DataSetSplitResult malignResult = dataSetSplit(inputsTemp, outputTemp, trainingPercent);

        Double[][] trainingInput = new Double[benignResult.trainingInput.length +
                malignResult.trainingInput.length][];
        Double[][] trainingOut = new Double[benignResult.trainingOutput.length +
                malignResult.trainingOutput.length][];
        Double[][] testingInput = new Double[benignResult.testingInput.length +
                malignResult.testingInput.length][];
        Double[][] testingOut = new Double[benignResult.testingOutput.length +
                malignResult.testingOutput.length][];

        int indexTraining = 0;
        for (int i = 0; i < benignResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = benignResult.trainingInput[i];
            trainingOut[indexTraining] = benignResult.trainingOutput[i];
            indexTraining++;
        }
        for (int i = 0; i < malignResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = malignResult.trainingInput[i];
            trainingOut[indexTraining] = malignResult.trainingOutput[i];
            indexTraining++;
        }

        int indexTesting = 0;
        for (int i = 0; i < benignResult.testingInput.length; i++) {
            testingInput[indexTesting] = benignResult.testingInput[i];
            testingOut[indexTesting] = benignResult.testingOutput[i];
            indexTesting++;
        }
        for (int i = 0; i < malignResult.testingInput.length; i++) {
            testingInput[indexTesting] = malignResult.testingInput[i];
            testingOut[indexTesting] = malignResult.testingOutput[i];
            indexTesting++;
        }

        return new DataSetSplitResult(trainingInput, trainingOut, testingInput, testingOut);
    }

    private static DataSetSplitResult dataSetSplit(Double[][] inputs, Double[][] outputs, double trainingPercent) {
        shuffle(inputs, outputs);

        trainingPercent = (trainingPercent < 0 || trainingPercent > 1) ? 0 : trainingPercent;

        int limit = (int) (inputs.length * trainingPercent);
        Double[][] trainingInput = new Double[limit][inputs[0].length];
        Double[][] trainingOutput = new Double[limit][outputs[0].length];

        int index = 0;
        for (; index < limit; index++) {
            trainingInput[index] = inputs[index];
            trainingOutput[index] = outputs[index];
        }

        Double[][] testingInput = new Double[inputs.length - limit][inputs[0].length];
        Double[][] testingOutput = new Double[outputs.length - limit][outputs[0].length];
        for (int i = 0; index < inputs.length; index++) {
            testingInput[i] = inputs[index];
            testingOutput[i] = outputs[index];
            i++;
        }

        return new DataSetSplitResult(trainingInput, trainingOutput, testingInput, testingOutput);
    }

    private static void shuffle(Double[][] inputs, Double[][] outputs) {
        final double percent = 0.5;
        Random r = new Random();
        for (int i = 0; i < inputs.length; i++) {
            if (r.nextGaussian() < percent) {
                // change
                final int index = r.nextInt(inputs.length);
                Double[] aux = inputs[i];
                inputs[i] = inputs[index];
                inputs[index] = aux;

                Double[] auxOut = outputs[i];
                outputs[i] = outputs[index];
                outputs[index] = auxOut;
            }
        }
    }

    private static class DataSetSplitResult {

        private Double[][] trainingInput;
        private Double[][] trainingOutput;
        private Double[][] testingInput;
        private Double[][] testingOutput;

        public DataSetSplitResult(Double[][] trainingInput, Double[][] trainingOutput, Double[][] testingInput, Double[][] testingOutput) {
            this.trainingInput = trainingInput;
            this.trainingOutput = trainingOutput;
            this.testingInput = testingInput;
            this.testingOutput = testingOutput;
        }
    }

    private static class ReadFileResult {

        private Double[][] inputs;
        private Double[][] targets;
        private Double[][] testingInput;
        private Double[][] testingOutput;

        public ReadFileResult(Double[][] inputs, Double[][] targets, Double[][] testingInput, Double[][] testingOutput) {
            this.inputs = inputs;
            this.targets = targets;
            this.testingInput = testingInput;
            this.testingOutput = testingOutput;
        }
    }

    private static class Sample {
        private Double[] inputs;
        private Double[] targets;

        public Sample(Double[] inputs, Double[] targets) {
            this.inputs = inputs;
            this.targets = targets;
        }

    }

}
