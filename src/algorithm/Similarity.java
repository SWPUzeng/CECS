package algorithm;

import java.util.*;

import static algorithm.CECS.*;

public class Similarity {

    public static void main(String[] args) {
        String filePath = "";
        List<List<Double>> numericData = readNumericDataFromFile(filePath);

        List<List<Double>> data = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        extractFeaturesAndLabels(numericData, data, labels);
        //double[][] standardizedData = standardNormalization(data);
        double[][] standardizedData = maxMinNormalization(data);


        double[][] X_train, X_test;
        double[] y_train, y_test;
        int fold = 5;
        int seed = 42;
        int testSize = (int) (standardizedData.length / fold);

        shuffleDataAndLabels(standardizedData, labels, seed);
        List<Double> accuracyList = new ArrayList<>();

        for (int l = 0; l < fold; l++) {
            System.out.println((l+1)+" fold");
            X_train = new double[standardizedData.length - testSize][standardizedData[0].length];
            X_test = new double[testSize][standardizedData[0].length];
            y_train = new double[standardizedData.length - testSize];
            y_test = new double[testSize];

            int startIndex = l * testSize;
            int endIndex = (l + 1) * testSize;
            for (int i = 0, j = 0; i < standardizedData.length; i++) {
                if (i >= startIndex && i < endIndex) {

                    System.arraycopy(standardizedData[i], 0, X_test[j], 0, standardizedData[i].length);
                    y_test[j] = labels.get(i);
                    j++;
                } else {

                    System.arraycopy(standardizedData[i], 0, X_train[i - j], 0, standardizedData[i].length);
                    y_train[i - j] = labels.get(i);
                }
            }

            List<List<double[]>> dataByLabel = groupDataByLabel(y_train, X_train);
            int f =1;
            for (List<double[]> labelData : dataByLabel) {
                System.out.println("label "+f);
                int numRows = labelData.size();
                int numCols = labelData.get(0).length;
                double[][] twoDArray = new double[numRows][numCols];
                for (int i = 0; i < numRows; i++) {
                    System.arraycopy(labelData.get(i), 0, twoDArray[i], 0, numCols);
                }

                int[][] resultall = discretization(twoDArray, standardizedData, 0.11);

                List<List<List<Integer>>> mergedConceptsall = new ArrayList<>();
                int numRowsall = resultall.length;
                int numColumnsall = resultall[0].length;

                for (int p = 0; p < numRowsall; p++) {
                    List<List<Integer>> conceptsall1 = getCommonProject(resultall, p);
                    mergeAndAdd(mergedConceptsall, conceptsall1);

                }

                for (int u = 0; u < numColumnsall; u++) {
                    List<List<Integer>> conceptsall2 = getCommonUser(resultall, u);
                    if (!conceptsall2.isEmpty()) {
                        mergeAndAdd(mergedConceptsall, conceptsall2);
                    }
                }
                calculateAndPrintSimilarities(mergedConceptsall);
                f++;
            }
        }



    }
    public static void calculateAndPrintSimilarities(List<List<List<Integer>>> mergedConcepts) {
        List<Double> similarities = new ArrayList<>();

        for (int i = 0; i < mergedConcepts.size(); i++) {
            for (int j = i + 1; j < mergedConcepts.size(); j++) {
                List<Integer> list1 = mergedConcepts.get(i).get(0);
                List<Integer> list2 = mergedConcepts.get(j).get(0);

                double jaccardSimilarity = calculateJaccardSimilarity(list1, list2);
                similarities.add(jaccardSimilarity);
                //System.out.println("Similarity between " + list1 + " and " + list2 + ": " + jaccardSimilarity);
            }
        }

        double minSimilarity = similarities.stream().min(Double::compare).orElse(0.0);
        double maxSimilarity = similarities.stream().max(Double::compare).orElse(0.0);
        double avgSimilarity = similarities.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        System.out.println("Min Similarity: " + minSimilarity);
        System.out.println("Max Similarity: " + maxSimilarity);
        System.out.println("Average Similarity: " + avgSimilarity);
    }

    public static double calculateJaccardSimilarity(List<Integer> list1, List<Integer> list2) {
        Set<Integer> set1 = new HashSet<>(list1);
        Set<Integer> set2 = new HashSet<>(list2);

        int intersectionSize = 0;
        int unionSize = set1.size() + set2.size();

        for (Integer element : set1) {
            if (set2.contains(element)) {
                intersectionSize++;
                unionSize--;
            }
        }

        return intersectionSize / (double) unionSize;
    }
}