import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Home {

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double[][] inputToHiddenWeights;
    private final double[][] hiddenToOutputWeights;
    private final double[] hiddenBiases;
    private final double[] outputBiases;
    private final Random random;



    public static void main(String[] args) {
        try {
            Reader reader = Files.newBufferedReader(Paths.get("datas.csv"));
            CSVParser parser = new CSVParserBuilder().withSeparator(',').build();
            CSVReader csvReader = new CSVReaderBuilder(reader)
                    .withCSVParser(parser)
                    .withSkipLines(1)
                    .build();

            List<String[]> data = csvReader.readAll();



            long seed = System.nanoTime();
            Collections.shuffle(data, new Random(seed));


            double splitRatio = 0.7;
            int splitIndex = (int) (data.size() * splitRatio);


            List<String[]> trainingData = data.subList(0, splitIndex);
            List<String[]> validationData = data.subList(splitIndex, data.size());


            List<double[]> trainingInputs = new ArrayList<>();
            List<double[]> trainingTargets = new ArrayList<>();
            List<double[]> validationInputs = new ArrayList<>();
            List<double[]> validationTargets = new ArrayList<>();

            for (String[] row : trainingData) {
                double[] input = new double[12];
                for (int i = 2; i < row.length; i++) {
                    input[i - 2] = Double.parseDouble(row[i]);
                }
                trainingInputs.add(input);

                double[] target = new double[3];
                String result = row[0];
                if ("H".equals(result)) {
                    target[0] = 0;
                    target[1] = 1;
                    target[2] = 0;
                } else if ("A".equals(result)) {
                    target[0] = 1;
                    target[1] = 0;
                    target[2] = 0;
                } else if ("D".equals(result)) {
                    target[0] = 0;
                    target[1] = 0;
                    target[2] = 1;
                }
                trainingTargets.add(target);
            }

            for (String[] row : validationData) {
                double[] input = new double[12];
                for (int i = 2; i < row.length; i++) {
                    input[i - 2] = Double.parseDouble(row[i]);
                }
                validationInputs.add(input);

                double[] target = new double[3];
                String result = row[0];
                if ("H".equals(result)) {
                    target[0] = 0;
                    target[1] = 1;
                    target[2] = 0;
                }else if ("A".equals(result)) {
                    target[0] = 1;
                    target[1] = 0;
                    target[2] = 0;
                }else if ("D".equals(result)) {
                    target[0] = 0;
                    target[1] = 0;
                    target[2] = 1;
                }
                validationTargets.add(target);
            }

            int inputSize = 12;             // Data Input
            int hiddenSize = 10;             // Neurons Number
            int outputSize = 3;             // Data Output
            int epochs = 200;                // Epochs Training
            double learningRate = 0.07;      // Learning Rate

            Home neuralNetwork = new Home(inputSize, hiddenSize, outputSize);
            neuralNetwork.train(trainingInputs.toArray(new double[0][0]), trainingTargets.toArray(new double[0][0]), epochs, learningRate);

            double validationError = neuralNetwork.validate(validationInputs.toArray(new double[0][0]), validationTargets.toArray(new double[0][0]));
            System.out.println("Validation Error: " + validationError);

            double[] inputToPredict = validationInputs.get(0);
            double[] prediction = neuralNetwork.forward(inputToPredict);
            System.out.println("Victory Prediction(Home Team: Away Team : Draw): " + Arrays.toString(prediction));


        } catch (IOException e) {
            System.out.println(e.getMessage());
        }

    }



    public Home(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        inputToHiddenWeights = new double[inputSize][hiddenSize];
        hiddenToOutputWeights = new double[hiddenSize][outputSize];
        hiddenBiases = new double[hiddenSize];
        outputBiases = new double[outputSize];
        random = new Random();
        initWeights();
    }




    public double validate(double[][] inputs, double[][] targets) {
        double totalError = 0.0;

        for (int i = 0; i < inputs.length; i++) {
            double[] output = forward(inputs[i]);
            double[] error = new double[outputSize];
            for (int k = 0; k < outputSize; k++) {
                error[k] = targets[i][k] - output[k];
            }
            double dataPointError = 0.0;
            for (int k = 0; k < outputSize; k++) {
                dataPointError += Math.pow(error[k], 2);
            }
            dataPointError /= outputSize;
            totalError += dataPointError;
        }

        return totalError / inputs.length;
    }

    public double[] forward(double[] input) {
        double[] hidden = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            hidden[i] = hiddenBiases[i];
            for (int j = 0; j < inputSize; j++) {
                hidden[i] += input[j] * inputToHiddenWeights[j][i];
            }
            hidden[i] = sigmoid(hidden[i]);
        }
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = outputBiases[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += hidden[j] * hiddenToOutputWeights[j][i];
            }
            output[i] = sigmoid(output[i]);
        }
        return output;
    }



    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        double totalError = 0.0;

        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < inputs.length; j++) {
                double[] output = forward(inputs[j]);
                double[] error = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    error[k] = targets[j][k] - output[k];
                }
                double dataPointError = 0.0;
                for (int k = 0; k < outputSize; k++) {
                    dataPointError += Math.pow(error[k], 2);
                }
                dataPointError /= outputSize;
                totalError += dataPointError;

                double averageError = totalError / inputs.length;
               // System.out.println("Epoch " + (i + 1) + " - Average Error: " + averageError);
                totalError = 0.0;

                double[] hiddenError = new double[hiddenSize];
                for (int k = 0; k < hiddenSize; k++) {
                    hiddenError[k] = 0;
                    for (int l = 0; l < outputSize; l++) {
                        hiddenError[k] += error[l] * hiddenToOutputWeights[k][l];
                    }
                }
                double[] hidden = new double[hiddenSize];
                for (int k = 0; k < outputSize; k++) {
                    outputBiases[k] += learningRate * error[k];
                    for (int l = 0; l < hiddenSize; l++) {
                        hiddenToOutputWeights[l][k] += learningRate * error[k] * hidden[l];
                    }
                }
                for (int k = 0; k < hiddenSize; k++) {
                    hiddenBiases[k] += learningRate * hiddenError[k];
                    for (int l = 0; l < inputSize; l++) {
                        inputToHiddenWeights[l][k] += learningRate * hiddenError[k] * inputs[j][l];
                    }
                }
            }
        }
    }



    private void initWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                inputToHiddenWeights[i][j] = random.nextGaussian();
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                hiddenToOutputWeights[i][j] = random.nextGaussian();
            }
        }
    }


    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));


    }
}
