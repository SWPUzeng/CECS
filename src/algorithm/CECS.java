package algorithm;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


public class CECS {

    static void shuffleDataAndLabels(double[][] data, List<Double> labels, int seed) {
        /*
         Method of randomly shuffling data sets and labels
        */

        List<Integer> indexList = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            indexList.add(i);
        }
        //To achieve repeatability, specify a random seed
        //long seed = 42;
        Random random = new Random(seed);
        Collections.shuffle(indexList,random);
        //Define scrambled data and labels
        double[][] shuffledData = new double[data.length][data[0].length];
        List<Double> shuffledLabels = new ArrayList<>();

        for (int i = 0; i < data.length; i++) {
            int originalIndex = indexList.get(i);
            System.arraycopy(data[originalIndex], 0, shuffledData[i], 0, data[originalIndex].length);
            shuffledLabels.add(labels.get(originalIndex));
        }
        /*
        //Print scrambled data sets
         for (int i = 0; i < shuffledData.length; i++) {
             System.out.print("Data: [");
            for (int j = 0; j < shuffledData[i].length; j++) {
               System.out.print(shuffledData[i][j]);
                if (j < shuffledData[i].length - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("], Label: " + shuffledLabels.get(i));
        }*/
        // Assign scrambled data sets and labels back to the original array and list
        System.arraycopy(shuffledData, 0, data, 0, shuffledData.length);
        labels.clear();
        labels.addAll(shuffledLabels);
    }

    public static double calculateAverage(List<Double> list) {
        /*
        calculate the average amount
        */

        double sum = 0;
        for (double value : list) {
            sum += value;
        }
        return roundToFourDecimalPlaces(sum / list.size());
    }

    public static double roundToFourDecimalPlaces(double value) {
         /*
        Reserved decimal place
         */
        double scaleFactor = Math.pow(10, 4);
        return Math.round(value * scaleFactor) / scaleFactor;
    }

    public static double calculateStandardDeviation(List<Double> list) {
        /*
        Calculated standard deviation
         */

        double mean = calculateAverage(list);
        double sumSquaredDiff = 0;
        for (double value : list) {
            sumSquaredDiff += Math.pow(value - mean, 2);
        }
        return roundToFourDecimalPlaces(Math.sqrt(sumSquaredDiff / list.size()));
    }

    public static List<List<Double>> readNumericDataFromFile(String filePath) {
        /*
        data fetch
         */

        List<List<Double>> numericData_in = new ArrayList<>();
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
                        System.out.println("Values that cannot be resolved: " + item);
                    }
                }
                if (!numericRow.isEmpty()) {
                    numericData_in.add(numericRow);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return numericData_in;
    }

    public static void extractFeaturesAndLabels(List<List<Double>> numericData,
                                                List<List<Double>> data, List<Double> labels) {
        /*
        Extract the features and labels from the entered numericData,
        then store them in two different lists: data and Labels
         */

        if (numericData == null || numericData.isEmpty()) {
            throw new IllegalArgumentException("numericData cannot be empty");
        }
        data.clear();
        labels.clear();

        for (List<Double> sublist : numericData) {
            int lastIndex = sublist.size() - 1;
            List<Double> features = sublist.subList(0, lastIndex);
            Double label = sublist.get(lastIndex);
            data.add(features);
            labels.add(label);
        }
    }

    public static double[][] standardNormalization(List<List<Double>> dataList) {
        /*
        Standard normalized function
         */

        if (dataList == null || dataList.isEmpty()) {
            throw new IllegalArgumentException("dataList cannot be empty");
        }

        int numRows = dataList.size();
        int numCols = dataList.get(0).size();
        double[][] data = new double[numRows][numCols];
        double[] means = new double[numCols];
        double[] stds = new double[numCols];

        for (int i = 0; i < numRows; i++) {
            List<Double> row = dataList.get(i);
            for (int j = 0; j < numCols; j++) {
                data[i][j] = row.get(j);
            }
        }
        for (int j = 0; j < numCols; j++) {
            double sum = 0.0;
            for (int i = 0; i < numRows; i++) {
                sum += data[i][j];
            }
            means[j] = sum / numRows;
        }
        for (int j = 0; j < numCols; j++) {
            double sumSquaredDiff = 0.0;
            for (int i = 0; i < numRows; i++) {
                double diff = data[i][j] - means[j];
                sumSquaredDiff += diff * diff;
            }
            stds[j] = Math.sqrt(sumSquaredDiff / numRows);
        }
        double[][] standardizedData = new double[numRows][numCols];
        DecimalFormat df = new DecimalFormat("#.00");

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double standardizedValue = (data[i][j] - means[j]) / stds[j];
                standardizedData[i][j] = Double.parseDouble(df.format(standardizedValue));
                //System.out.println(standardizedData[i][j]);
            }
        }
        return standardizedData;
    }

    public static double[][] maxMinNormalization(List<List<Double>> dataList) {
        /*
        Max-min normalization function
         */

        if (dataList == null || dataList.isEmpty()) {
            throw new IllegalArgumentException("dataList cannot be empty");
        }
        int numRows = dataList.size();
        int numCols = dataList.get(0).size();
        double[][] normalizedData = new double[numRows][numCols];

        double[] minValues = new double[numCols];
        double[] maxValues = new double[numCols];

        for (int j = 0; j < numCols; j++) {
            double minVal = Double.MAX_VALUE;
            double maxVal = Double.MIN_VALUE;
            for (List<Double> doubles : dataList) {
                double value = doubles.get(j);
                if (value < minVal) {
                    minVal = value;
                }
                if (value > maxVal) {
                    maxVal = value;
                }
            }
            minValues[j] = minVal;
            maxValues[j] = maxVal;
        }
        //System.out.println(numRows);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double value = dataList.get(i).get(j);
                normalizedData[i][j] = (value - minValues[j]) / (maxValues[j] - minValues[j]);
                //System.out.print(normalizedData[i][j]+" ");
            }
            //System.out.println("\n");
        }
        return normalizedData;
    }

    public static void splitDataIntoTrainAndTest(double[][] standardizedData, List<Double> labels,
                                                 double testRatio, double[][] X_train, double[][] X_test,
                                                 double[] y_train, double[] y_test, long seed) {
        /*
        Divide the training set and test set
         */

        if (standardizedData == null || labels == null || standardizedData.length != labels.size()) {
            throw new IllegalArgumentException("The entered data is invalid");
        }

        int dataSize = standardizedData.length;
        Random random = new Random(seed);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < dataSize; i++) {
            indices.add(i);
        }
        for (int i = indices.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, temp);
        }
        int testSize = (int) (dataSize * testRatio);

        for (int i = 0; i < dataSize; i++) {
            double[] row = standardizedData[indices.get(i)];

            if (i < testSize) {
                X_test[i] = row;
                y_test[i] = labels.get(indices.get(i));
            } else {
                X_train[i - testSize] = row;
                y_train[i - testSize] = labels.get(indices.get(i));
            }
        }
    }

    public static List<List<double[]>> groupDataByLabel(double[] y_train, double[][] X_train) {
        /*
        Get data with the same label, that is, the sub-formal context
         */

        if (y_train == null || X_train == null || y_train.length != X_train.length) {
            throw new IllegalArgumentException("The entered data is invalid");
        }

        Set<Double> uniqueLabels = new HashSet<>();
        for (double label : y_train) {
            uniqueLabels.add(label);
        }

        List<List<double[]>> dataByLabel = new ArrayList<>();
        for (int i = 0; i < uniqueLabels.size(); i++) {
            dataByLabel.add(new ArrayList<>());
        }

        for (int i = 0; i < y_train.length; i++) {
            double label = y_train[i];
            double[] labelData = X_train[i];
            int labelIndex = (int) label;
            dataByLabel.get(labelIndex).add(labelData);
        }

        return dataByLabel;
    }

    public static double[] calculateMetrics(int numLabels,List<Integer> y_true,List<Integer> y_predict){
        /*
        Accuracy and Macro_F1 calculation
         */

        int[] TP = new int[numLabels];
        int[] FP = new int[numLabels];
        int[] FN = new int[numLabels];

        double[] precision = new double[numLabels];
        double[] recall = new double[numLabels];
        double[] F1 = new double[numLabels];

        for (int i = 0; i < y_true.size(); i++) {
            int trueLabel = y_true.get(i);
            int predLabel = y_predict.get(i);

            if (trueLabel == predLabel) {
                TP[trueLabel]++;
            } else {
                FP[predLabel]++;
                FN[trueLabel]++;
            }
        }

        for (int label = 0; label < numLabels; label++) {
            if (TP[label] + FP[label] == 0) {
                precision[label] = 0.0;
            } else {
                precision[label] = (double) TP[label] / (TP[label] + FP[label]);
            }

            if (TP[label] + FN[label] == 0) {
                recall[label] = 0.0;
            } else {
                recall[label] = (double) TP[label] / (TP[label] + FN[label]);
            }

            if (precision[label] + recall[label] == 0) {
                F1[label] = 0.0;
            } else {
                F1[label] = 2.0 * (precision[label] * recall[label]) / (precision[label] + recall[label]);
            }
        }

        double accuracy = (double) sum(TP) / y_true.size();
        //double average_precision = mean(precision);
        //double average_recall = mean(recall);
        double Macro_F1 = mean(F1);

        DecimalFormat decimalFormat = new DecimalFormat("0.0000");
        String formattedAccuracy = decimalFormat.format(accuracy);
        String formattedMacro_F1 = decimalFormat.format(Macro_F1);

        return new double[]{Double.parseDouble(formattedAccuracy), Double.parseDouble(formattedMacro_F1)};
    }

    public static int sum(int[] arr) {
        /*
        Calculate sum value
         */

        int total = 0;
        for (int value : arr) {
            total += value;
        }
        return total;
    }

    public static double mean(double[] arr) {
        /*
        Calculate mean value
         */

        double total = 0.0;
        for (double value : arr) {
            total += value;
        }
        return total / arr.length;
    }

    public static List<algorithm.Range> generateRanges(double minVal, double maxVal, double width) {
        /*
        This function is used to generate small intervals during discretization.
         */

        List<algorithm.Range> ranges = new ArrayList<>();
        double start = minVal;
        while (roundTooneDecimal(start) < maxVal) {
            double end = Math.min(start + width, maxVal);
            ranges.add(new algorithm.Range(roundToThreeDecimal(start), roundToThreeDecimal(end)));
            start += width;
        }
        return ranges;
    }

    private static double roundTooneDecimal(double value) {
        return Math.round(value * 10.0) / 10.0;
    }
    
    public static int[][] discretization(double[][] partialData, double[][] standardizedData, double k) {
        /*
        This function discretizes the original data set into binary data
         */

        //计算每列的最大值与最小值
        double[] minValues = calculateMinValues(standardizedData);
        double[] maxValues = calculateMaxValues(standardizedData);


        List<List<algorithm.Range>> rangesList = new ArrayList<>();
        for (int i = 0; i < minValues.length; i++) {
            rangesList.add(generateRanges(minValues[i], maxValues[i], k));
        }
        /*
        Print the generated interval
        for (List<sample.Range> outerList : rangesList) {
            for (sample.Range range : outerList) {
                System.out.print("( " + range.start + ", " + range.end+")");
            }
        }
         */
        int numColumns = partialData[0].length;
        int[][] result = new int[partialData.length][calculateTotalRanges(rangesList)];
        int columnStartIdx = 0;
        for (int column = 0; column < numColumns; column++) {
            double[] columnData = getColumnData(partialData, column);
            List<algorithm.Range> ranges = rangesList.get(column);
            int numRanges = ranges.size();
            int columnEndIdx = columnStartIdx + numRanges;

            for (int i = 0; i < columnData.length; i++) {
                double value = columnData[i];
                boolean valueAssigned = false;
                for (int j = 0; j < numRanges; j++) {
                    algorithm.Range range = ranges.get(j);
                    if(value == range.end){
                        result[i][columnStartIdx + j] = 1;
                    }
                    if (range.start <= value && value < range.end) {
                        result[i][columnStartIdx + j] = 1;
                        valueAssigned = true;
                        break;
                    }
                }
                if (!valueAssigned && columnStartIdx > 0) {
                    result[i][columnStartIdx - 1] = 1;
                }
            }
            columnStartIdx = columnEndIdx;
        }
        return result;
    }

    public static double roundToThreeDecimal(double value) {
        return Math.round(value * 100.0) / 100.0;
    }

    public static double[] getColumnData(double[][] data, int columnIndex) {
        /*
        Get the value of a column
         */
        int numRows = data.length;
        double[] columnData = new double[numRows];
        for (int i = 0; i < numRows; i++) {
            columnData[i] = data[i][columnIndex];
        }
        return columnData;
    }

    public static int calculateTotalRanges(List<List<algorithm.Range>> rangesList) {
        int totalRanges = 0;
        for (List<Range> ranges : rangesList) {
            totalRanges += ranges.size();
        }
        //System.out.println(totalRanges);
        return totalRanges;
    }

    public static double jaccardSimilarity(Set<Integer> set1, Set<Integer> set2) {
        /*
        Calculate the jaccard similarity
         */

        Set<Integer> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        Set<Integer> union = new HashSet<>(set1);
        union.addAll(set2);
        double intersectionSize = intersection.size();
        double unionSize = union.size();
        return (unionSize != 0) ? (intersectionSize / unionSize) : 0;
    }

    public static double calculatePrediction(List<List<List<Integer>>> concepts, Set<Integer> targetSet, int dataNumber) {
        /*
        Label confidence value calculation formula
         */

        double totalSum = 0;
        int numSublists = concepts.size();

        for (List<List<Integer>> sublist : concepts) {
            Set<Integer> set1 = new HashSet<>(sublist.get(0));
            Set<Integer> set2 = new HashSet<>(sublist.get(1));
            double weight = set1.size();

            double jaccard = jaccardSimilarity(set2, targetSet);
            double weightedJaccard = weight * jaccard;
            totalSum += weightedJaccard;
        }
        return (numSublists != 0) ? ((totalSum / numSublists)/(dataNumber-1)) : 0;
    }

    public static double[] calculateMinValues(double[][] data) {
        /*
        Get the minimum value for each column in the data set
         */

        int numCols = data[0].length;
        double[] minValues = new double[numCols];

        for (int j = 0; j < numCols; j++) {
            double min = Double.POSITIVE_INFINITY;
            for (double[] datum : data) {
                if (datum[j] < min) {
                    min = datum[j];
                }
            }
            minValues[j] = min;
        }

        return minValues;
    }

    public static double[] calculateMaxValues(double[][] data) {
        /*
        Get the maximum value for each column in the data set
         */
        int numCols = data[0].length;
        double[] maxValues = new double[numCols];

        for (int j = 0; j < numCols; j++) {
            double max = Double.NEGATIVE_INFINITY;
            for (double[] datum : data) {
                if (datum[j] > max) {
                    max = datum[j];
                }
            }
            maxValues[j] = max;
        }
        return maxValues;
    }

    public static void mergeAndAdd(List<List<List<Integer>>> mergedConcepts, List<List<Integer>> concepts) {
        /*
        Integrate all concepts into mergedConcepts and remove the same concepts
         */

        boolean isDuplicate = false;
        List<Set<Integer>> conceptsSet = new ArrayList<>();

        for (List<Integer> subList : concepts) {
            conceptsSet.add(new HashSet<>(subList));
        }
        for (List<List<Integer>> existingConceptGroup : mergedConcepts) {
            boolean allSublistsMatch = true;

            if (existingConceptGroup.size() == conceptsSet.size()) {
                for (List<Integer> existingSublist : existingConceptGroup) {
                    boolean sublistMatches = false;

                    for (Set<Integer> newSublistSet : conceptsSet) {
                        if (newSublistSet.equals(new HashSet<>(existingSublist))) {
                            sublistMatches = true;
                            break;
                        }
                    }
                    if (!sublistMatches) {
                        allSublistsMatch = false;
                        break;
                    }
                }
            } else {
                allSublistsMatch = false;
            }
            if (allSublistsMatch) {
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            mergedConcepts.add(concepts);
        }
    }

    public static List<List<Integer>> getCommonUser(int[][] formalContext, int listProjectIndex) {
        /*
        Generate concepts through attributes
         */

        List<List<Integer>> concept = new ArrayList<>();
        Set<Integer> userSet = new HashSet<>();

        for (int i = 0; i < formalContext.length; i++) {
            if (formalContext[i][listProjectIndex] == 1) {
                userSet.add(i);
            }
        }
        if (userSet.isEmpty()) {
            return concept;
        }
        List<Integer> projectSet = new ArrayList<>();

        for (int projectIndex = 0; projectIndex < formalContext[0].length; projectIndex++) {
            Set<Integer> projectUsers = new HashSet<>();

            for (int i = 0; i < formalContext.length; i++) {
                if (formalContext[i][projectIndex] == 1) {
                    projectUsers.add(i);
                }
            }

            if (projectUsers.containsAll(userSet)) {
                projectSet.add(projectIndex);
            }
        }
        List<Integer> list = new ArrayList<>(userSet);
        concept.add(list);
        concept.add(projectSet);

        return concept;
    }

    public static List<List<Integer>> getCommonProject(int[][] formalContext, int listUserIndex) {
        /*
        Generate concepts through objects
         */

        List<List<Integer>> concept = new ArrayList<>();
        Set<Integer> projectSet = new HashSet<>();

        for (int i = 0; i < formalContext[listUserIndex].length; i++) {
            if (formalContext[listUserIndex][i] == 1) {
                projectSet.add(i);
            }
        }
        List<Integer> userSet = new ArrayList<>();

        for (int userIndex = 0; userIndex < formalContext.length; userIndex++) {
            Set<Integer> userProjects = new HashSet<>();

            // 获取每个用户的项目集合
            for (int i = 0; i < formalContext[userIndex].length; i++) {
                if (formalContext[userIndex][i] == 1) {
                    userProjects.add(i);
                }
            }
            if (userProjects.containsAll(projectSet)) {
                userSet.add(userIndex);
            }
        }
        concept.add(userSet);
        List<Integer> list1 = new ArrayList<>(projectSet);
        concept.add(list1);

        return concept;
    }

    private static void printResult(List<double[]> resultList) {
        /*
        Find the index corresponding to the largest averageAccuracy value and print
        */

        double maxAccuracy = -1;
        int maxIndex = -1;

        for (int i = 0; i < resultList.size(); i++) {
            double currentAccuracy = resultList.get(i)[0];
            if (currentAccuracy > maxAccuracy) {
                maxAccuracy = currentAccuracy;
                maxIndex = i;
            }
        }
        double[] maxResult = resultList.get(maxIndex);
        System.out.println("Accuracy: " + maxResult[0] + " ±" + maxResult[1]);
        System.out.println("Macro_F1: " + maxResult[2] + " ±" + maxResult[3]);
    }


    public static void main(String[] args) {
        String filePath = "";
        List<List<Double>> numericData = readNumericDataFromFile(filePath);
        List<List<Double>> data = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        extractFeaturesAndLabels(numericData, data, labels);
        //double[][] standardizedData = standardNormalization(data);
        double[][] standardizedData = maxMinNormalization(data);
        double k = 0.1;
        List<double[]> resultList = new ArrayList<>();
        while (k <= 0.33) {
            double[][] X_train, X_test;
            double[] y_train, y_test;
            int fold = 5;
            int seed = 42;
            int testSize = (int) (standardizedData.length / fold);

            shuffleDataAndLabels(standardizedData, labels, seed);

            List<Double> accuracyList = new ArrayList<>();
            List<Double> Macro_F1List = new ArrayList<>();

            for (int l = 0; l < fold; l++) {

                X_train = new double[standardizedData.length - testSize][standardizedData[0].length];
                X_test = new double[testSize][standardizedData[0].length];
                y_train = new double[standardizedData.length - testSize];
                y_test = new double[testSize];

                int startIndex = l * testSize;
                int endIndex = (l + 1) * testSize;
                for (int i = 0, j = 0; i < standardizedData.length; i++) {
                    if (i >= startIndex && i < endIndex) {
                        // 加入到测试集
                        System.arraycopy(standardizedData[i], 0, X_test[j], 0, standardizedData[i].length);
                        y_test[j] = labels.get(i);
                        j++;
                    } else {
                        // 加入到训练集
                        System.arraycopy(standardizedData[i], 0, X_train[i - j], 0, standardizedData[i].length);
                        y_train[i - j] = labels.get(i);
                    }
                }

               /*
                 System.out.println("Fold " + (l + 1) + " Test Set:");
                 for (int i = 0; i < X_test.length; i++) {
                    System.out.println("Features: " + Arrays.toString(X_test[i]) + ", Label: " + y_test[i]);
                  }
                */

                List<List<double[]>> dataByLabel = groupDataByLabel(y_train, X_train);
                Set<Double> uniqueLabels = new HashSet<>();
                for (double label : y_train) {
                    uniqueLabels.add(label);
                }

                List<List<List<List<Integer>>>> allmergedConcepts = new ArrayList<>();

                for (List<double[]> labelData : dataByLabel) {
                    int numRows = labelData.size();
                    int numCols = labelData.get(0).length;
                    double[][] twoDArray = new double[numRows][numCols];

                    for (int i = 0; i < numRows; i++) {
                        for (int j = 0; j < numCols; j++) {
                            twoDArray[i][j] = labelData.get(i)[j];
                        }
                    }
                    int[][] result = discretization(twoDArray, standardizedData, k);
                    List<List<List<Integer>>> mergedConcepts = new ArrayList<>();

                    int numRows1 = result.length;
                    int numColumns = result[0].length;

                    for (int i = 0; i < numRows1; i++) {
                        List<List<Integer>> concepts1 = getCommonProject(result, i);
                        mergeAndAdd(mergedConcepts, concepts1);

                    }
                    for (int i = 0; i < numColumns; i++) {
                        List<List<Integer>> concepts2 = getCommonUser(result, i);
                        if (!concepts2.isEmpty()) {
                            mergeAndAdd(mergedConcepts, concepts2);
                        }
                    }
                    allmergedConcepts.add(mergedConcepts);

                }
                List<Integer> y_predict = new ArrayList<>();
                List<Integer> y_true = new ArrayList<>();

                for (double[] rowData : X_test) {
                    double[][] rowArray = new double[1][];
                    rowArray[0] = rowData;
                    int[][] result = discretization(rowArray, standardizedData, k);

                    Set<Integer> indicesSet = new HashSet<>();
                    for (int[] row2 : result) {
                        Set<Integer> indices1 = new HashSet<>();
                        for (int i = 0; i < row2.length; i++) {
                            if (row2[i] == 1) {
                                indices1.add(i);
                            }
                        }
                        indicesSet = indices1;
                    }
                    List<Double> predictions = new ArrayList<>();
                    for (List<List<List<Integer>>> conceptList : allmergedConcepts) {
                        double prediction = calculatePrediction(conceptList, indicesSet, numericData.size());
                        predictions.add(prediction);
                    }
                    double maxPrediction = Double.MIN_VALUE;
                    int maxPredictionIndex = -1;
                    for (int i = 0; i < predictions.size(); i++) {
                        double prediction = predictions.get(i);

                        if (prediction > maxPrediction) {
                            maxPrediction = prediction;
                            maxPredictionIndex = i;
                        }
                    }
                    y_predict.add(maxPredictionIndex);
                }
                for (int i = 0; i < X_test.length; i++) {
                    y_true.add((int) y_test[i]);
                }
                int numLabels = uniqueLabels.size();

                double[] result = calculateMetrics(numLabels, y_true, y_predict);

                accuracyList.add(result[0]);
                Macro_F1List.add(result[1]);
            }

            double averageAccuracy = calculateAverage(accuracyList);
            double stdDevAccuracy = calculateStandardDeviation(accuracyList);
            //Macro_F1
            double averageMacro_F1 = calculateAverage(Macro_F1List);
            double stdDevMacro_F1 = calculateStandardDeviation(Macro_F1List);

            double[] result = {averageAccuracy, stdDevAccuracy, averageMacro_F1, stdDevMacro_F1};
            resultList.add(result);
            k += 0.01;
        }
        printResult(resultList);

    }


}







