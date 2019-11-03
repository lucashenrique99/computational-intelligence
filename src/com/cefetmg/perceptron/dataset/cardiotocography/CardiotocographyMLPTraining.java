package com.cefetmg.perceptron.dataset.cardiotocography;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;
import com.cefetmg.perceptron.utils.FileUtils;
import com.cefetmg.perceptron.utils.InputFunctions;
import com.cefetmg.perceptron.utils.Optimizations;
import com.cefetmg.perceptron.utils.XYLineChart;
import com.cefetmg.perceptron.utils.model.DataSetSplitResult;
import com.cefetmg.perceptron.utils.model.ReadFileResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;


public class CardiotocographyMLPTraining {

    public static void main(String[] args) {
//        singleLearningCoefficient();
        dynamicLearningCoefficient();
    }

    public static void singleLearningCoefficient() {

        ReadFileResult readFileResult = readAllDataSetFile();
        Double[][] source = readFileResult.inputs;
        Double[][] target = readFileResult.targets;

        MultiLayerPerceptron normalPerceptron = new MultiLayerPerceptron(21, 9, 1);
        MultiLayerPerceptron suspectPerceptron = new MultiLayerPerceptron(21, 9, 1);
        MultiLayerPerceptron pathologicalPerceptron = new MultiLayerPerceptron(21, 9, 1);

        final Double learningCoefficient = 0.000001d;

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

                Double[] normalResult = normalPerceptron.training(input, new Double[]{output[0]}, learningCoefficient);
                Double[] suspectResult = suspectPerceptron.training(input, new Double[]{output[1]}, learningCoefficient);
                Double[] pathologicalResult = pathologicalPerceptron.training(input, new Double[]{output[2]}, learningCoefficient);

                Optimizations.thresholdTruncate(normalResult, suspectResult, pathologicalResult);
//                Optimizations.thresholdChangeRange(normalResult, suspectResult, pathologicalResult);

                double sampleError = 0;
                int sampleErrorClassifier = 0;

                // normal perceptron
                Double normalError = Math.abs(normalResult[0] - output[0]); // error
                sampleError += normalError;
                Double normalOutValue = Optimizations.thresholdErrorTruncate(normalResult[0]); // classifier error
                sampleErrorClassifier += Math.abs(output[0] - normalOutValue);

                // suspect perceptron
                Double suspectError = Math.abs(suspectResult[0] - output[1]);// error
                sampleError += suspectError;
                Double suspectOutValue = Optimizations.thresholdErrorTruncate(suspectResult[0]); // classifier error
                sampleErrorClassifier += Math.abs(output[1] - suspectOutValue);

                // pathological perceptron
                Double pathologicalError = Math.abs(pathologicalResult[0] - output[2]);// error
                sampleError += pathologicalError;
                Double pathologicalOutValue = Optimizations.thresholdErrorTruncate(pathologicalResult[0]); // classifier error
                sampleErrorClassifier += Math.abs(output[2] - pathologicalOutValue);

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
            values[0][epoch] = new Double[]{(double) epoch, ((double) epochErrorClassifier / sampleTrainingIndex)};

            int sampleTestingIndex = 0;
            double epochTestingError = 0;
            int epochErrorTestingClassifier = 0;
            for (; sampleTestingIndex < readFileResult.testingInput.length; sampleTestingIndex++) {
                double sampleTestingError = 0;
                Double[] out1 = normalPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out2 = suspectPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out3 = pathologicalPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);

                if (Math.round(out1[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][0]) ||
                        Math.round(out2[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][1]) ||
                        Math.round(out3[0]) != Math.round(readFileResult.testingOutput[sampleTestingIndex][2])) {
                    epochErrorTestingClassifier++;
                }

                // normal perceptron
                Double normalError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]); // error
                sampleTestingError += normalError;

                // suspect perceptron
                Double suspectError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][1] - out2[0]);// error
                sampleTestingError += suspectError;

                // pathological perceptron
                Double pathologicalError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][2] - out3[0]);// error
                sampleTestingError += pathologicalError;

                epochTestingError += sampleTestingError;
            }

            System.out.print("\tError Testing: " + epochTestingError);
            System.out.print("\tError Testing classifier: " + epochErrorTestingClassifier);
            values[1][epoch] = new Double[]{(double) epoch, ((double) epochErrorTestingClassifier / sampleTestingIndex)};
        }

        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "normalPerceptron", normalPerceptron);
        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "suspectPerceptron", suspectPerceptron);
        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "pathologicalPerceptron", pathologicalPerceptron);

        XYLineChart.showChart("Cardiotocography Training", new String[]{"Training", "Testing"}, values);
        System.out.println();
        System.out.println("Classifier -> \t min " + minEpochClassifier + " \t max " + maxEpochClassifier);
        System.out.println("Error -> \t min " + minEpochError + " \t max " + maxEpochError);

    }

    public static void dynamicLearningCoefficient() {

//        ReadFileResult readFileResult = readAllDataSetFile();
        ReadFileResult readFileResult = readDividedDataSetsFiles();
//        ReadFileResult readFileResult = readDividedBalancedDataSetsFiles();
        Double[][] source = readFileResult.inputs;
        Double[][] target = readFileResult.targets;

//        final int intermediateNeurons = 12;
        final int intermediateNeurons = 6;

        MultiLayerPerceptron normalPerceptron = new MultiLayerPerceptron(21, intermediateNeurons, 1);
        MultiLayerPerceptron normalPerceptron1 = new MultiLayerPerceptron(21, intermediateNeurons, 1);
        MultiLayerPerceptron normalPerceptron2 = new MultiLayerPerceptron(21, intermediateNeurons, 1);

        MultiLayerPerceptron suspectPerceptron = new MultiLayerPerceptron(21, intermediateNeurons, 1);
        MultiLayerPerceptron suspectPerceptron1 = new MultiLayerPerceptron(21, intermediateNeurons, 1);
        MultiLayerPerceptron suspectPerceptron2 = new MultiLayerPerceptron(21, intermediateNeurons, 1);

        MultiLayerPerceptron pathologicalPerceptron = new MultiLayerPerceptron(21, intermediateNeurons, 1);
        MultiLayerPerceptron pathologicalPerceptron1 = new MultiLayerPerceptron(21, intermediateNeurons, 1);
        MultiLayerPerceptron pathologicalPerceptron2 = new MultiLayerPerceptron(21, intermediateNeurons, 1);

        final double learningCoefficientVariation = 0.3;
        final double minLearningCoefficient = Math.pow(2, -10);
        double normalLearningCoefficient = 0.08d;
        double suspectLearningCoefficient = 0.08d;
        double pathologicalLearningCoefficient = 0.08d;

        double minEpochClassifier = Double.MAX_VALUE;
        double maxEpochClassifier = 0;

        double minEpochError = Double.MAX_VALUE;
        double maxEpochError = 0;

        final int numEpochs = 5000;
        final int epochPrintInterval = 100;
        String print;
        Double[][][] values = new Double[2][][];
        values[0] = new Double[numEpochs][2]; // training
        values[1] = new Double[numEpochs][2]; // testing
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double epochError = 0;
            int epochErrorClassifier = 0;
            int sampleTrainingIndex = 0;
            boolean retry = true;
            for (; sampleTrainingIndex < source.length; sampleTrainingIndex++) {

                Double[] input = source[sampleTrainingIndex];
                Double[] output = target[sampleTrainingIndex]; // [ 1 0 0] | [0 0 1] | [0 1 0]

                final double normalLearningCoefficient1 = normalLearningCoefficient * (1 + learningCoefficientVariation);
                final double normalLearningCoefficient2 = normalLearningCoefficient * (1 - learningCoefficientVariation);
                final double suspectLearningCoefficient1 = suspectLearningCoefficient * (1 + learningCoefficientVariation);
                final double suspectLearningCoefficient2 = suspectLearningCoefficient * (1 - learningCoefficientVariation);
                final double pathologicalLearningCoefficient1 = pathologicalLearningCoefficient * (1 + learningCoefficientVariation);
                final double pathologicalLearningCoefficient2 = pathologicalLearningCoefficient * (1 - learningCoefficientVariation);

                Double[] normalResult1 = normalPerceptron1.training(input, new Double[]{output[0]}, normalLearningCoefficient1);
                Double[] normalResult2 = normalPerceptron2.training(input, new Double[]{output[0]}, normalLearningCoefficient2);
                Double[] suspectResult1 = suspectPerceptron1.training(input, new Double[]{output[1]}, suspectLearningCoefficient1);
                Double[] suspectResult2 = suspectPerceptron2.training(input, new Double[]{output[1]}, suspectLearningCoefficient2);
                Double[] pathologicalResult1 = pathologicalPerceptron1.training(input, new Double[]{output[2]}, pathologicalLearningCoefficient1);
                Double[] pathologicalResult2 = pathologicalPerceptron2.training(input, new Double[]{output[2]}, pathologicalLearningCoefficient2);

                Optimizations.thresholdTruncate(normalResult1, suspectResult1, pathologicalResult1);
                Optimizations.thresholdTruncate(normalResult2, suspectResult2, pathologicalResult2);
//                Optimizations.thresholdChangeRange(normalResult1, suspectResult1, pathologicalResult1);
//                Optimizations.thresholdChangeRange(normalResult2, suspectResult2, pathologicalResult2);

                final boolean isNormal1Better = (Math.abs(normalResult1[0] - output[0]) < Math.abs(normalResult2[0] - output[0]));
                final boolean isSuspect1Better = (Math.abs(suspectResult1[0] - output[1]) < Math.abs(suspectResult2[0] - output[1]));
                final boolean isPathological1Better = (Math.abs(pathologicalResult1[0] - output[2]) < Math.abs(pathologicalResult2[0] - output[2]));

                final Double[] normalResult;
                if (isNormal1Better) {
                    normalResult = normalResult1;

                    final Double[][] intermediate = normalPerceptron1.getIntermediateWeights();
                    final Double[][] out = normalPerceptron1.getOutputWeights();

                    normalPerceptron.setIntermediateWeights(intermediate);
                    normalPerceptron.setOutputWeights(out);

                    normalPerceptron2.setIntermediateWeights(intermediate);
                    normalPerceptron2.setOutputWeights(out);

                    normalLearningCoefficient = (normalLearningCoefficient1 < minLearningCoefficient) ? normalLearningCoefficient : normalLearningCoefficient1;

                } else {
                    normalResult = normalResult2;

                    final Double[][] intermediate = normalPerceptron2.getIntermediateWeights();
                    final Double[][] out = normalPerceptron2.getOutputWeights();

                    normalPerceptron.setIntermediateWeights(intermediate);
                    normalPerceptron.setOutputWeights(out);

                    normalPerceptron1.setIntermediateWeights(intermediate);
                    normalPerceptron1.setOutputWeights(out);

                    normalLearningCoefficient = (normalLearningCoefficient2 < minLearningCoefficient) ? normalLearningCoefficient : normalLearningCoefficient2;
                }

                final Double[] suspectResult;
                if (isSuspect1Better) {
                    suspectResult = suspectResult1;

                    final Double[][] intermediate = suspectPerceptron1.getIntermediateWeights();
                    final Double[][] out = suspectPerceptron1.getOutputWeights();

                    suspectPerceptron.setIntermediateWeights(intermediate);
                    suspectPerceptron.setOutputWeights(out);

                    suspectPerceptron2.setIntermediateWeights(intermediate);
                    suspectPerceptron2.setOutputWeights(out);

                    suspectLearningCoefficient = (suspectLearningCoefficient1 < minLearningCoefficient) ? suspectLearningCoefficient : suspectLearningCoefficient1;

                } else {
                    suspectResult = suspectResult2;

                    final Double[][] intermediate = suspectPerceptron2.getIntermediateWeights();
                    final Double[][] out = suspectPerceptron2.getOutputWeights();

                    suspectPerceptron.setIntermediateWeights(intermediate);
                    suspectPerceptron.setOutputWeights(out);

                    suspectPerceptron1.setIntermediateWeights(intermediate);
                    suspectPerceptron1.setOutputWeights(out);

                    suspectLearningCoefficient = (suspectLearningCoefficient2 < minLearningCoefficient) ? suspectLearningCoefficient : suspectLearningCoefficient2;
                }

                final Double[] pathologicalResult;
                if (isPathological1Better) {
                    pathologicalResult = pathologicalResult1;

                    final Double[][] intermediate = pathologicalPerceptron1.getIntermediateWeights();
                    final Double[][] out = pathologicalPerceptron1.getOutputWeights();

                    pathologicalPerceptron.setIntermediateWeights(intermediate);
                    pathologicalPerceptron.setOutputWeights(out);

                    pathologicalPerceptron2.setIntermediateWeights(intermediate);
                    pathologicalPerceptron2.setOutputWeights(out);

                    pathologicalLearningCoefficient = (pathologicalLearningCoefficient1 < minLearningCoefficient) ? pathologicalLearningCoefficient : pathologicalLearningCoefficient1;

                } else {
                    pathologicalResult = pathologicalResult2;

                    final Double[][] intermediate = pathologicalPerceptron2.getIntermediateWeights();
                    final Double[][] out = pathologicalPerceptron2.getOutputWeights();

                    pathologicalPerceptron.setIntermediateWeights(intermediate);
                    pathologicalPerceptron.setOutputWeights(out);

                    pathologicalPerceptron1.setIntermediateWeights(intermediate);
                    pathologicalPerceptron1.setOutputWeights(out);

                    pathologicalLearningCoefficient = (pathologicalLearningCoefficient2 < minLearningCoefficient) ? pathologicalLearningCoefficient : pathologicalLearningCoefficient2;
                }

                double sampleError = 0;

                // normal perceptron
                Double normalError = Math.abs(normalResult[0] - output[0]); // error
                sampleError += normalError;

                // suspect perceptron
                Double suspectError = Math.abs(suspectResult[0] - output[1]);// error
                sampleError += suspectError;

                // pathological perceptron
                Double pathologicalError = Math.abs(pathologicalResult[0] - output[2]);// error
                sampleError += pathologicalError;

                final int errorTemp = Math.min(1,
                      (int) (Math.abs( output[0] - normalResult[0]) +
                                Math.abs( output[1] - suspectResult[0]) +
                                Math.abs( output[2] - pathologicalResult[0])));

//                if(errorTemp == 1){
//                    if(retry){
//                        retry = false;
//                        sampleTrainingIndex--;
//                        continue;
//                    } else {
//                        retry = true;
//                    }
//                }

                // group normal, suspect and pathological errors
                epochError += sampleError;
                epochErrorClassifier += errorTemp;
            }

            minEpochClassifier = Math.min(minEpochClassifier, epochErrorClassifier);
            maxEpochClassifier = Math.max(maxEpochClassifier, epochErrorClassifier);
            minEpochError = Math.min(minEpochError, epochError);
            maxEpochError = Math.max(maxEpochError, epochError);

//            if(epoch % epochPrintInterval == 0) {
//                System.out.print("\nEPOCH " + epoch + ": ");
//                System.out.print("\tError Training: " + epochError);
//                System.out.print("\tError Training classifier: " + epochErrorClassifier);
//            }


            values[0][epoch] = new Double[]{(double) epoch, ((double) epochErrorClassifier / source.length)};

            int sampleTestingIndex = 0;
            double epochTestingError = 0;
            int epochErrorTestingClassifier = 0;
            for (; sampleTestingIndex < readFileResult.testingInput.length; sampleTestingIndex++) {
                double sampleTestingError = 0;
                Double[] out1 = normalPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out2 = suspectPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
                Double[] out3 = pathologicalPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);

                epochErrorTestingClassifier += Math.min(1,
                        Math.abs( readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]) +
                                Math.abs( readFileResult.testingOutput[sampleTestingIndex][1] - out2[0]) +
                                Math.abs( readFileResult.testingOutput[sampleTestingIndex][2] - out3[0]));

                // normal perceptron
                Double normalError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]); // error
                sampleTestingError += normalError;

                // suspect perceptron
                Double suspectError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][1] - out2[0]);// error
                sampleTestingError += suspectError;

                // pathological perceptron
                Double pathologicalError = Math.abs(readFileResult.testingOutput[sampleTestingIndex][2] - out3[0]);// error
                sampleTestingError += pathologicalError;

                epochTestingError += sampleTestingError;
            }

//            if(epoch % epochPrintInterval == 0) {
//                System.out.print("\tError Testing: " + epochTestingError);
//                System.out.print("\tError Testing classifier: " + epochErrorTestingClassifier);
//            }

            values[1][epoch] = new Double[]{(double) epoch, ((double) epochErrorTestingClassifier / readFileResult.testingInput.length)};

            if (epoch % epochPrintInterval == 0) {
                print = "\nEPOCH " + epoch + ": " + "\tError Training: " + epochError + "\tError Training classifier: " + epochErrorClassifier;
                print += "\tError Testing: " + epochTestingError + "\tError Testing classifier: " + epochErrorTestingClassifier;
                System.out.println(print);
            }
        }


        XYLineChart.showChart("Cardiotocography Training", new String[]{"Training", "Testing"}, values);
        System.out.println();
        System.out.println("Classifier -> \t min " + minEpochClassifier + " \t max " + maxEpochClassifier);
        System.out.println("Error -> \t min " + minEpochError + " \t max " + maxEpochError);

        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "normalPerceptron5", normalPerceptron);
        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "suspectPerceptron5", suspectPerceptron);
        FileUtils.saveObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "pathologicalPerceptron5", pathologicalPerceptron);

        System.out.println("Saved Object");
    }


    /*
        output codifications ->
            N -> 1 0 0
            S -> 0 1 0
            P -> 0 0 1
     */
    public static ReadFileResult readAllDataSetFile() {
        try {
            InputStreamReader inputStreamReaderInputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/allInputs.txt")));
            BufferedReader bufferedReaderInputs = new BufferedReader(inputStreamReaderInputs);
            InputStreamReader inputStreamReaderOutputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/allOutputs.txt")));
            BufferedReader bufferedReaderOutputs = new BufferedReader(inputStreamReaderOutputs);

            int datasetSize = 2126;

            String line;
            String outputLine;
            Double[][] inputs = new Double[datasetSize][];
            Double[][] outputs = new Double[datasetSize][];
            int index = 0;
            while ((line = bufferedReaderInputs.readLine()) != null) {
                outputLine = bufferedReaderOutputs.readLine();

                String[] inputVector = line.split("\\s+");
                Double[] input = new Double[inputVector.length];
                for (int i = 0; i < inputVector.length; i++) {
                    input[i] = Double.parseDouble(inputVector[i]);
                }

                String[] outputVector = outputLine.split("\\s+");
                Double[] output = new Double[outputVector.length];
                for (int i = 0; i < outputVector.length; i++) {
                    output[i] = Double.parseDouble(outputVector[i]);
                }

                inputs[index] = input;
                outputs[index] = output;

                index++;
            }

            Double[][] testingInput = new Double[inputs.length][inputs[0].length];
            Double[][] testingOutput = new Double[outputs.length][outputs[0].length];

            // if the data set will be divide, this lines below should be used
            DataSetSplitResult dataSetSplitResult = InputFunctions.dataSetSplit(inputs, outputs, 0.75d);
            inputs = dataSetSplitResult.trainingInput;
            outputs = dataSetSplitResult.trainingOutput;
            testingInput = dataSetSplitResult.testingInput;
            testingOutput = dataSetSplitResult.testingOutput;


            normalize(inputs);
            normalize(testingInput);

            return new ReadFileResult(inputs, outputs, testingInput, testingOutput);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static ReadFileResult readDividedDataSetsFiles() {
        try {
            InputStreamReader inputStreamReaderInputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTreino_in.txt")));
            BufferedReader bufferedReaderInputs = new BufferedReader(inputStreamReaderInputs);

            InputStreamReader inputStreamReaderOutputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTreino_out.txt")));
            BufferedReader bufferedReaderOutputs = new BufferedReader(inputStreamReaderOutputs);

            InputStreamReader inputStreamReaderInputsTest = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTeste_in.txt")));
            BufferedReader bufferedReaderInputsTest = new BufferedReader(inputStreamReaderInputsTest);

            InputStreamReader inputStreamReaderOutputsTest = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTeste_out.txt")));
            BufferedReader bufferedReaderOutputsTest = new BufferedReader(inputStreamReaderOutputsTest);

            final int datasetSize = 1591;

            String line;
            String outputLine;
            Double[][] inputs = new Double[datasetSize][];
            Double[][] outputs = new Double[datasetSize][];
            int index = 0;
            while ((line = bufferedReaderInputs.readLine()) != null) {
                outputLine = bufferedReaderOutputs.readLine();

                String[] inputVector = line.split("\\s+");
                Double[] input = new Double[inputVector.length];
                for (int i = 0; i < inputVector.length; i++) {
                    input[i] = Double.parseDouble(inputVector[i]);
                }

                String[] outputVector = outputLine.split("\\s+");
                Double[] output = new Double[outputVector.length];
                for (int i = 0; i < outputVector.length; i++) {
                    output[i] = Double.parseDouble(outputVector[i]);
                }

                inputs[index] = input;
                outputs[index] = output;

                index++;
            }

            normalize(inputs);

            final int datasetTestSize = 535;
            Double[][] testingInput = new Double[datasetTestSize][];
            Double[][] testingOutput = new Double[datasetTestSize][];
            index = 0;
            while ((line = bufferedReaderInputsTest.readLine()) != null) {
                outputLine = bufferedReaderOutputsTest.readLine();

                String[] inputVector = line.split("\\s+");
                Double[] input = new Double[inputVector.length];
                for (int i = 0; i < inputVector.length; i++) {
                    input[i] = Double.parseDouble(inputVector[i]);
                }

                String[] outputVector = outputLine.split("\\s+");
                Double[] output = new Double[outputVector.length];
                for (int i = 0; i < outputVector.length; i++) {
                    output[i] = Double.parseDouble(outputVector[i]);
                }

                testingInput[index] = input;
                testingOutput[index] = output;

                index++;
            }

            normalize(testingInput);

            return new ReadFileResult(inputs, outputs, testingInput, testingOutput);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static ReadFileResult readDividedBalancedDataSetsFiles() {
        try {
            InputStreamReader inputStreamReaderInputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTreinoBalanced_in.txt")));
            BufferedReader bufferedReaderInputs = new BufferedReader(inputStreamReaderInputs);

            InputStreamReader inputStreamReaderOutputs = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTreinoBalanced_out.txt")));
            BufferedReader bufferedReaderOutputs = new BufferedReader(inputStreamReaderOutputs);

            InputStreamReader inputStreamReaderInputsTest = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTeste_in.txt")));
            BufferedReader bufferedReaderInputsTest = new BufferedReader(inputStreamReaderInputsTest);

            InputStreamReader inputStreamReaderOutputsTest = new InputStreamReader(new FileInputStream(new File("src/com/cefetmg/perceptron/dataset/cardiotocography/files/BaseTeste_out.txt")));
            BufferedReader bufferedReaderOutputsTest = new BufferedReader(inputStreamReaderOutputsTest);

            final int datasetSize = 3608;

            String line;
            String outputLine;
            Double[][] inputs = new Double[datasetSize][];
            Double[][] outputs = new Double[datasetSize][];
            int index = 0;
            while ((line = bufferedReaderInputs.readLine()) != null) {
                outputLine = bufferedReaderOutputs.readLine();

                String[] inputVector = line.split("\\s+");
                Double[] input = new Double[inputVector.length];
                for (int i = 0; i < inputVector.length; i++) {
                    input[i] = Double.parseDouble(inputVector[i]);
                }

                String[] outputVector = outputLine.split("\\s+");
                Double[] output = new Double[outputVector.length];
                for (int i = 0; i < outputVector.length; i++) {
                    output[i] = Double.parseDouble(outputVector[i]);
                }

                inputs[index] = input;
                outputs[index] = output;

                index++;
            }

            normalize(inputs);

            final int datasetTestSize = 535;
            Double[][] testingInput = new Double[datasetTestSize][];
            Double[][] testingOutput = new Double[datasetTestSize][];
            index = 0;
            while ((line = bufferedReaderInputsTest.readLine()) != null) {
                outputLine = bufferedReaderOutputsTest.readLine();

                String[] inputVector = line.split("\\s+");
                Double[] input = new Double[inputVector.length];
                for (int i = 0; i < inputVector.length; i++) {
                    input[i] = Double.parseDouble(inputVector[i]);
                }

                String[] outputVector = outputLine.split("\\s+");
                Double[] output = new Double[outputVector.length];
                for (int i = 0; i < outputVector.length; i++) {
                    output[i] = Double.parseDouble(outputVector[i]);
                }

                testingInput[index] = input;
                testingOutput[index] = output;

                index++;
            }

            normalize(testingInput);

            return new ReadFileResult(inputs, outputs, testingInput, testingOutput);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void normalize(Double[][] input) {
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                switch (j) {
                    case 0: // interval 106 ~ 160
                        input[i][j] = (input[i][j] - 106) / 54;
                        break;
                    case 1:
                    case 3:
                    case 4:
                        input[i][j] = input[i][j] * 10;
                        break;
                    case 5:
                    case 6:
                        input[i][j] = input[i][j] * 100;
                        break;
                    case 7: // interval 12 ~ 87
                        input[i][j] = (input[i][j] - 12) / 75;
                        break;
                    case 8: // interval 0.2 ~ 7
                        input[i][j] = (input[i][j] - 0.2) / 6.8;
                        break;
                    case 9: // interval 0 ~ 91
                        input[i][j] = input[i][j] / 91;
                        break;
                    case 10: // interval 0 ~ 50.7
                        input[i][j] = input[i][j] / 50.7;
                        break;
                    case 11: // interval 3 ~ 180
                        input[i][j] = (input[i][j] - 3) / 177;
                        break;
                    case 12: // interval 50 ~ 159
                        input[i][j] = (input[i][j] - 50) / 109;
                        break;
                    case 13: // interval 122 ~ 238
                        input[i][j] = (input[i][j] - 122) / 116;
                        break;
                    case 14: // interval 0 ~ 18
                        input[i][j] = input[i][j]/ 18;
                        break;
                    case 15: // interval 0 ~ 10
                        input[i][j] = input[i][j]/ 10;
                        break;
                    case 16: // interval 60 ~ 187
                        input[i][j] = (input[i][j] - 60) / 127;
                        break;
                    case 17: // interval 73 ~ 182
                        input[i][j] = (input[i][j] - 73) / 109;
                        break;
                    case 18: // interval 77 ~ 186
                        input[i][j] = (input[i][j] - 77) / 109;
                        break;
                    case 19: // interval 0 ~ 269
                        input[i][j] = input[i][j] / 269;
                        break;
                }
            }
        }
    }

}
