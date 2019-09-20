package com.cefetmg.perceptron.utils;

import com.cefetmg.perceptron.model.MultiLayerPerceptron;

public class Optimizations {

    /*
    * Function to truncate vector.
    * Put 1 on the index that value is bigger comparing in this array, and zero otherwise
    */
    public static void thresholdTruncate(Double[] vector){
        int index = 0;
        double value = 0;
        for (int i = 0; i < vector.length; i++) {
            if(value < vector[i]){
                value = vector[i];
                index = i;
            }
            vector[i] = 0d;
        }

        vector[index] = 1d;
    }

    /*
    * Function to truncate vector.
    * Put 1 on the index that value is bigger comparing in this array, and zero otherwise
    */
    public static void thresholdTruncate(Double[] ...vectors){
        int index = 0;
        double value = 0;
        for (int i = 0; i < vectors.length; i++) {
            if(value < vectors[i][0]){
                value = vectors[i][0];
                index = i;
            }
            vectors[i][0] = 0d;
        }

        vectors[index][0] = 1d;
    }

    public static void thresholdChangeRange(Double[] ...vectors){
        final double acceptedErrorRange = 0.05;
        for (int i = 0; i < vectors.length; i++) {
            if(vectors[i][0] < acceptedErrorRange){
                vectors[i][0] = 0d;
            } else if(vectors[i][0] > (1 - acceptedErrorRange)){
                vectors[i][0] = 1d;
            }
        }
    }

    public static double thresholdErrorTruncate(Double value){
        final double threshold = 0.5;
        return (value < threshold) ? 0 : 1d;
    }

}
