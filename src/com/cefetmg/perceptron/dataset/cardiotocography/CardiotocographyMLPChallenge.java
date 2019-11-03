package com.cefetmg.perceptron.dataset.cardiotocography;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;
import com.cefetmg.perceptron.utils.FileUtils;
import com.cefetmg.perceptron.utils.model.ReadFileResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class CardiotocographyMLPChallenge {

    public static void main(String[] args) {

        ReadFileResult readFileResult = readDividedDataSetsFiles();
        Double[][] source = readFileResult.inputs;
        Double[][] target = readFileResult.targets;

        MultiLayerPerceptron normalPerceptron = (MultiLayerPerceptron) FileUtils.openObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "normalPerceptron4");
        MultiLayerPerceptron suspectPerceptron = (MultiLayerPerceptron) FileUtils.openObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "suspectPerceptron4");
        MultiLayerPerceptron pathologicalPerceptron = (MultiLayerPerceptron) FileUtils.openObject("src/com/cefetmg/perceptron/dataset/cardiotocography/savedObjects/", "pathologicalPerceptron4");

        double epochError = 0;
        int epochErrorClassifier = 0;
        int sampleTrainingIndex = 0;
        for (; sampleTrainingIndex < source.length; sampleTrainingIndex++) {

            Double[] input = source[sampleTrainingIndex];
            Double[] output = target[sampleTrainingIndex]; // [ 1 0 0] | [0 0 1] | [0 1 0]

            Double[] normalResult = normalPerceptron.getTargetBySource(input);
            Double[] suspectResult = suspectPerceptron.getTargetBySource(input);
            Double[] pathologicalResult = pathologicalPerceptron.getTargetBySource(input);

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

            // group normal, suspect and pathological errors
            epochError += sampleError;
            epochErrorClassifier += Math.min(1,
                    Math.abs(output[0] - normalResult[0]) +
                            Math.abs(output[1] - suspectResult[0]) +
                            Math.abs(output[2] - pathologicalResult[0]));
        }

        System.out.print("\tError Training: " + epochError);
        System.out.print("\tError Training classifier: " + epochErrorClassifier);
        System.out.print("\tError Training Percent: " + ((double) epochErrorClassifier / source.length));
        System.out.println();

        int sampleTestingIndex = 0;
        double epochTestingError = 0;
        int epochErrorTestingClassifier = 0;
        for (; sampleTestingIndex < readFileResult.testingInput.length; sampleTestingIndex++) {
            double sampleTestingError = 0;
            Double[] out1 = normalPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
            Double[] out2 = suspectPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);
            Double[] out3 = pathologicalPerceptron.getTargetBySource(readFileResult.testingInput[sampleTestingIndex]);

            epochErrorTestingClassifier += Math.min(1,
                    Math.abs(readFileResult.testingOutput[sampleTestingIndex][0] - out1[0]) +
                            Math.abs(readFileResult.testingOutput[sampleTestingIndex][1] - out2[0]) +
                            Math.abs(readFileResult.testingOutput[sampleTestingIndex][2] - out3[0]));

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
        System.out.print("\tError Testing Percent: " + ((double) epochErrorTestingClassifier / readFileResult.testingInput.length));
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

            CardiotocographyMLPTraining.normalize(inputs);
            CardiotocographyMLPTraining.normalize(testingInput);

            return new ReadFileResult(inputs, outputs, testingInput, testingOutput);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }


}
