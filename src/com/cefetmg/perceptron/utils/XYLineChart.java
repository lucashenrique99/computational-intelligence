package com.cefetmg.perceptron.utils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class XYLineChart extends JFrame {

    private XYLineChart(String title, String[] legend, Double[][]... values) {
        super(title);
        // Create dataset
        XYDataset dataset = createDataset(legend,values);

        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                "Epoch",
                "Error",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        // Create Panel
        ChartPanel panel = new ChartPanel(chart);
        setContentPane(panel);
    }

    private XYDataset createDataset(String[] legend, Double[][]... values) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int j = 0; j < values.length; j++) {
            XYSeries series = new XYSeries(legend[j]);
            for (int i = 0; i < values[0].length; i++) {
                series.add(values[j][i][0], values[j][i][1]);
            }
            //Add series to dataset
            dataset.addSeries(series);
        }

        return dataset;
    }

    public static XYLineChart showChart(String title, String[] legend, Double[][]... values) {
        XYLineChart chat = new XYLineChart(title, legend,values);
        chat.setSize(800, 400);
        chat.setLocationRelativeTo(null);
        chat.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        chat.setVisible(true);

        return chat;
    }
}
