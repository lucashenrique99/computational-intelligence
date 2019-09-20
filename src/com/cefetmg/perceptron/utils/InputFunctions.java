package com.cefetmg.perceptron.utils;

import com.cefetmg.perceptron.utils.model.DataSetSplitResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class InputFunctions {

    public static DataSetSplitResult dataSetSplit(Double[][] inputs, Double[][] outputs, final double trainingPercent) {

        ArrayList<Double[]> normalInput = new ArrayList<>();
        List<Double[]> normalOutput = new ArrayList<>();
        List<Double[]> suspectInput = new ArrayList<>();
        List<Double[]> suspectOutput = new ArrayList<>();
        List<Double[]> pathologicalInput = new ArrayList<>();
        List<Double[]> pathologicalOutput = new ArrayList<>();
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i][0] == 1) {
                normalInput.add(inputs[i]);
                normalOutput.add(outputs[i]);
            } else if (outputs[i][1] == 1) {
                suspectInput.add(inputs[i]);
                suspectOutput.add(outputs[i]);
            } else if (outputs[i][2] == 1) {
                pathologicalInput.add(inputs[i]);
                pathologicalOutput.add(outputs[i]);
            } else {
                System.out.println("ERROR");
            }
        }

        Double[][] inputsTemp = new Double[normalInput.size()][];
        Double[][] outputTemp = new Double[normalInput.size()][];
        for (int i = 0; i < normalInput.size(); i++) {
            inputsTemp[i] = normalInput.get(i);
            outputTemp[i] = normalOutput.get(i);
        }
        DataSetSplitResult normalResult = classDataSetSplit(inputsTemp, outputTemp, trainingPercent);

        inputsTemp = new Double[suspectInput.size()][];
        outputTemp = new Double[suspectOutput.size()][];
        for (int i = 0; i < suspectInput.size(); i++) {
            inputsTemp[i] = suspectInput.get(i);
            outputTemp[i] = suspectOutput.get(i);
        }
        DataSetSplitResult suspectResult = classDataSetSplit(inputsTemp, outputTemp, trainingPercent);

        inputsTemp = new Double[pathologicalInput.size()][];
        outputTemp = new Double[pathologicalOutput.size()][];
        for (int i = 0; i < pathologicalInput.size(); i++) {
            inputsTemp[i] = pathologicalInput.get(i);
            outputTemp[i] = pathologicalOutput.get(i);
        }
        DataSetSplitResult pathologicalResult = classDataSetSplit(inputsTemp, outputTemp, trainingPercent);

        Double[][] trainingInput = new Double[normalResult.trainingInput.length +
                suspectResult.trainingInput.length +
                pathologicalResult.trainingInput.length][];
        Double[][] trainingOut = new Double[normalResult.trainingOutput.length +
                suspectResult.trainingOutput.length +
                pathologicalResult.trainingOutput.length][];
        Double[][] testingInput = new Double[normalResult.testingInput.length +
                suspectResult.testingInput.length +
                pathologicalResult.testingInput.length][];
        Double[][] testingOut = new Double[normalResult.testingOutput.length +
                suspectResult.testingOutput.length +
                pathologicalResult.testingOutput.length][];

        int indexTraining = 0;
        for (int i = 0; i < normalResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = normalResult.trainingInput[i];
            trainingOut[indexTraining] = normalResult.trainingOutput[i];
            indexTraining++;
        }
        for (int i = 0; i < suspectResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = suspectResult.trainingInput[i];
            trainingOut[indexTraining] = suspectResult.trainingOutput[i];
            indexTraining++;
        }
        for (int i = 0; i < pathologicalResult.trainingInput.length; i++) {
            trainingInput[indexTraining] = pathologicalResult.trainingInput[i];
            trainingOut[indexTraining] = pathologicalResult.trainingOutput[i];
            indexTraining++;
        }

        int indexTesting = 0;
        for (int i = 0; i < normalResult.testingInput.length; i++) {
            testingInput[indexTesting] = normalResult.testingInput[i];
            testingOut[indexTesting] = normalResult.testingOutput[i];
            indexTesting++;
        }
        for (int i = 0; i < suspectResult.testingInput.length; i++) {
            testingInput[indexTesting] = suspectResult.testingInput[i];
            testingOut[indexTesting] = suspectResult.testingOutput[i];
            indexTesting++;
        }
        for (int i = 0; i < pathologicalResult.testingInput.length; i++) {
            testingInput[indexTesting] = pathologicalResult.testingInput[i];
            testingOut[indexTesting] = pathologicalResult.testingOutput[i];
            indexTesting++;
        }


        return new DataSetSplitResult(trainingInput, trainingOut, testingInput, testingOut);
    }

    private static DataSetSplitResult classDataSetSplit(Double[][] inputs, Double[][] outputs, double trainingPercent) {
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

}
