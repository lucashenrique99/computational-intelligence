package com.cefetmg.perceptron_training.utils;

public class MathFunctions {

    public static Double sigmoidal(Double value){
        return 1/ (1 + Math.exp(-value));
    }

}
