package algorithm;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class MeanRank {

    public static void main(String[] args) {
        String filePath = "";
        List<List<Double>> numericData = readNumericDataFromFile(filePath);

        if (numericData.isEmpty() || numericData.get(0).isEmpty()) {
            System.out.println("None");
            return;
        }

        int rowCount = numericData.size();
        int columnCount = numericData.get(0).size();
        double[][] values = new double[rowCount][columnCount];

        for (int i = 0; i < rowCount; i++) {
            List<Double> row = numericData.get(i);
            for (int j = 0; j < columnCount; j++) {
                values[i][j] = row.get(j);
                System.out.print(values[i][j] + " ");
            }
            System.out.println();
        }
        calculateMeanRank(values);
    }

    private static void calculateMeanRank(double[][] values) {
        int rowCount = values.length;
        int colCount = values[0].length;

        double[][] rankMatrix = new double[rowCount][colCount];

        for (int i = 0; i < rowCount; i++) {
            double[] rowValues = values[i];
            int[] rankArray = calculateRank(rowValues);
            for (int j = 0; j < colCount; j++) {
                rankMatrix[i][j] = rankArray[j];
                System.out.print(rankMatrix[i][j]+" ");
            }
            System.out.println();
        }

        double[] meanRank = new double[colCount];

        for (int j = 0; j < colCount; j++) {
            double sumRank = 0;
            for (int i = 0; i < rowCount; i++) {
                sumRank += rankMatrix[i][j];
            }
            meanRank[j] = sumRank / rowCount;
        }

        for (int j = 0; j < colCount; j++) {
            System.out.printf("%.2f\t", meanRank[j]);
        }
    }

    private static int[] calculateRank(double[] rowValues) {
        int length = rowValues.length;
        int[] rankArray = new int[length];

        for (int i = 0; i < length; i++) {
            int rank = 1;
            for (int j = 0; j < length; j++) {
                if (j != i && rowValues[j] > rowValues[i]) {
                    rank++;
                }
            }
            rankArray[i] = rank;
            //System.out.println(rankArray[i]);
        }
        return rankArray;
    }

    public static List<List<Double>> readNumericDataFromFile(String filePath) {
        List<List<Double>> numericData = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] items = line.split("\t");
                List<Double> numericRow = new ArrayList<>();
                for (String item : items) {
                    try {
                        double numericValue = Double.parseDouble(item);
                        numericRow.add(numericValue);
                    } catch (NumberFormatException e) {
                        System.out.println("Cannot read: " + item);
                    }
                }
                if (!numericRow.isEmpty()) {
                    numericData.add(numericRow);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return numericData;
    }
}