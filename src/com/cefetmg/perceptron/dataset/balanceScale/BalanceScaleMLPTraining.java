package com.cefetmg.perceptron.dataset.balanceScale;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;
import com.cefetmg.perceptron.model.Perceptron;
import com.cefetmg.perceptron.utils.XYLineChart;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BalanceScaleMLPTraining {

    public static void main(String[] args) {

        ReadFileResult readFileResult = readInputFile(true);
        Double[][] source = readFileResult.inputs;
        Double[][] target = readFileResult.targets;

        MultiLayerPerceptron leftPerceptron = new MultiLayerPerceptron(4, 4,1);
        MultiLayerPerceptron balancePerceptron = new MultiLayerPerceptron(4, 4,1);
        MultiLayerPerceptron rightPerceptron = new MultiLayerPerceptron(4, 4,1);

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

                Double[] leftResult = leftPerceptron.training(input, new Double[]{output[0]}, learningCoefficient);
                Double[] balanceResult = balancePerceptron.training(input, new Double[]{output[1]}, learningCoefficient);
                Double[] rightResult = rightPerceptron.training(input, new Double[]{output[2]}, learningCoefficient);

                double sampleError = 0;
                int sampleErrorClassifier = 0;

                // left perceptron
                Double leftError = Math.abs(leftResult[0] - output[0]); // error
                sampleError += leftError;
                Double leftOutValue = (leftResult[0] < threshold) ? 0 : 1d; // classifier error
                sampleErrorClassifier += Math.abs(output[0] - leftOutValue);

                // balance perceptron
                Double balanceError = Math.abs(balanceResult[0] - output[1]);// error
                sampleError += balanceError;
                Double balanceOutValue = (balanceResult[0] < threshold) ? 0 : 1d; // classifier error
                sampleErrorClassifier += Math.abs(output[1] - balanceOutValue);

                // right perceptron
                Double rightError = Math.abs(rightResult[0] - output[2]);// error
                sampleError += rightError;
                Double rightOutValue = (rightResult[0] < threshold) ? 0 : 1d; // classifier error
                sampleErrorClassifier += Math.abs(output[2] - rightOutValue);

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
                Double[] out1 = leftPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out2 = balancePerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out3 = rightPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);

                if (Math.round(out1[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][0]) ||
                        Math.round(out2[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][1]) ||
                        Math.round(out3[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][2])) {
                    epochErrorTestingClassifier++;
                }

                // left perceptron
                Double leftError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]); // error
                sampleTestingError += leftError;

                // balance perceptron
                Double balanceError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][1] - out2[0]);// error
                sampleTestingError += balanceError;

                // right perceptron
                Double rightError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][2] - out3[0]);// error
                sampleTestingError += rightError;

                epochTestingError += sampleTestingError;
            }
//            System.out.println("Error Testing: " + epochTestingError);
//            System.out.println("Error Testing classifier: " + epochErrorTestingClassifier + "\t Percent: " + ((double) epochErrorTestingClassifier / sampleTestingIndex));
            System.out.print("\tError Testing: " + epochTestingError);
            System.out.print("\tError Testing classifier: " + epochErrorTestingClassifier);
            values[1][epoch] = new Double[]{(double) epoch, ((double) epochErrorTestingClassifier / sampleTestingIndex)};
        }

        XYLineChart.showChart("Balance Scale Training", new String[]{"Training", "Testing"}, values);
        System.out.println();
        System.out.println("Classifier -> \t min " + minEpochClassifier + " \t max " + maxEpochClassifier);
        System.out.println("Error -> \t min " + minEpochError + " \t max " + maxEpochError);

//        for (int i = 0; i < readFileResult.testingInput.length; i++) {
//            System.out.println("================ TEST " + i + " ====================");
//            System.out.println(Arrays.toString(readFileResult.testingOutput[i]));
//            testResults(readFileResult.testingInput[i], leftPerceptron, balancePerceptron, rightPerceptron);
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
            InputStreamReader ir = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/balanceScale/files/balance-scale.data")));
            BufferedReader in = new BufferedReader(ir);

            int datasetSize = 625;

            String line;
            Double[][] inputs = new Double[datasetSize][];
            Double[][] outputs = new Double[datasetSize][];
            int index = 0;
            while ((line = in.readLine()) != null) {

                String[] vector = line.split(",");

                String className = vector[0];

                Double[] output = new Double[]{
                        (className.equalsIgnoreCase("L")) ? 1d : 0,
                        (className.equalsIgnoreCase("B")) ? 1d : 0,
                        (className.equalsIgnoreCase("R")) ? 1d : 0,
                };

                String leftWeight = vector[1];
                String leftDistance = vector[2];
                String rightWeight = vector[3];
                String rightDistance = vector[4];
                Double[] input = new Double[]{
                        Double.parseDouble(leftWeight),
                        Double.parseDouble(leftDistance),
                        Double.parseDouble(rightWeight),
                        Double.parseDouble(rightDistance)
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

        ArrayList<Double[]> leftInput = new ArrayList<>();
        List<Double[]> leftOutput = new ArrayList<>();
        List<Double[]> balanceInput = new ArrayList<>();
        List<Double[]> balanceOutput = new ArrayList<>();
        List<Double[]> rightInput = new ArrayList<>();
        List<Double[]> rightOutput = new ArrayList<>();
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i][0] == 1) {
                leftInput.add(inputs[i]);
                leftOutput.add(outputs[i]);
            } else if (outputs[i][1] == 1) {
                balanceInput.add(inputs[i]);
                balanceOutput.add(outputs[i]);
            } else if (outputs[i][2] == 1) {
                rightInput.add(inputs[i]);
                rightOutput.add(outputs[i]);
            } else {
                System.out.println("ERROR");
            }
        }

        Double[][] inputsTemp = new Double[leftInput.size()][];
        Double[][] outputTemp = new Double[leftInput.size()][];
        for (int i = 0; i < leftInput.size(); i++) {
            inputsTemp[i] = leftInput.get(i);
            outputTemp[i] = leftOutput.get(i);
        }
        DataSetSplitResult leftResult = dataSetSplit(inputsTemp, outputTemp, trainingPercent);

        inputsTemp = new Double[balanceInput.size()][];
        outputTemp = new Double[balanceOutput.size()][];
        for (int i = 0; i < balanceInput.size(); i++) {
            inputsTemp[i] = balanceInput.get(i);
            outputTemp[i] = balanceOutput.get(i);
        }
        DataSetSplitResult balanceResult = dataSetSplit(inputsTemp, outputTemp, trainingPercent);

        inputsTemp = new Double[rightInput.size()][];
        outputTemp = new Double[rightOutput.size()][];
        for (int i = 0; i < rightInput.size(); i++) {
            inputsTemp[i] = rightInput.get(i);
            outputTemp[i] = rightOutput.get(i);
        }
        DataSetSplitResult rightResult = dataSetSplit(inputsTemp, outputTemp, trainingPercent);

        Double[][] trainingInput = new Double[leftResult.trainingInput.length +
                balanceResult.trainingInput.length +
                rightResult.trainingInput.length][];
        Double[][] trainingOut = new Double[leftResult.trainingOutput.length +
                balanceResult.trainingOutput.length +
                rightResult.trainingOutput.length][];
        Double[][] testingInput = new Double[leftResult.testingInput.length +
                balanceResult.testingInput.length +
                rightResult.testingInput.length][];
        Double[][] testingOut = new Double[leftResult.testingOutput.length +
                balanceResult.testingOutput.length +
                rightResult.testingOutput.length][];

        int indexTraining = 0;
        for (int i = 0; i < leftResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = leftResult.trainingInput[i];
            trainingOut[indexTraining] = leftResult.trainingOutput[i];
            indexTraining++;
        }
        for (int i = 0; i < balanceResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = balanceResult.trainingInput[i];
            trainingOut[indexTraining] = balanceResult.trainingOutput[i];
            indexTraining++;
        }
        for (int i = 0; i < rightResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = rightResult.trainingInput[i];
            trainingOut[indexTraining] = rightResult.trainingOutput[i];
            indexTraining++;
        }

        int indexTesting = 0;
        for (int i = 0; i < leftResult.testingInput.length; i++) {
            testingInput[indexTesting] = leftResult.testingInput[i];
            testingOut[indexTesting] = leftResult.testingOutput[i];
            indexTesting++;
        }
        for (int i = 0; i < balanceResult.testingInput.length; i++) {
            testingInput[indexTesting] = balanceResult.testingInput[i];
            testingOut[indexTesting] = balanceResult.testingOutput[i];
            indexTesting++;
        }
        for (int i = 0; i < rightResult.testingInput.length; i++) {
            testingInput[indexTesting] = rightResult.testingInput[i];
            testingOut[indexTesting] = rightResult.testingOutput[i];
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
