import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Home {

    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] inputToHiddenWeights;
    private double[][] hiddenToOutputWeights;
    private double[] hiddenBiases;
    private double[] outputBiases;
    private Random random;

    public static void main(String[] args) {
        try {
            Reader reader = Files.newBufferedReader(Paths.get("data.csv"));
            CSVParser parser = new CSVParserBuilder().withSeparator(',').build();
            CSVReader csvReader = new CSVReaderBuilder(reader)
                    .withCSVParser(parser)
                    .withSkipLines(1)
                    .build();

            List<String[]> data = csvReader.readAll();

            List<double[]> inputs = new ArrayList<>();
            List<double[]> targets = new ArrayList<>();

            for (String[] row : data) {
                double[] input = new double[16]; // Definir inputSize com o número correto de características
                for (int i = 2; i < row.length; i++) {
                    input[i - 2] = Double.parseDouble(row[i]);
                }
                inputs.add(input);

                double[] target = new double[3]; // Definir outputSize com o número correto de saídas

                // Mapear "H", "A" e "D" para valores numéricos
                String result = row[1];
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

                targets.add(target);
            }


            int inputSize = 16; // Tamanho  de entrada
            int hiddenSize = 10; // Tamanho da camada oculta
            int outputSize = 3; // Tamanho de saída
            int epochs = 1000; // Número de épocas de treinamento
            double learningRate = 0.01; // Taxa de aprendizado

            // Treinar a rede neural
            Home neuralNetwork = new Home(inputSize, hiddenSize, outputSize);
            neuralNetwork.train(inputs.toArray(new double[0][0]), targets.toArray(new double[0][0]), epochs, learningRate);

            // Fazer previsões usando a rede neural
            double[] inputToPredict = inputs.get(0); // Substitua pelo dado de entrada que você deseja prever
            double[] prediction = neuralNetwork.forward(inputToPredict);

            // Imprimir a previsão
            System.out.println("Previsão da rede neural: " + Arrays.toString(prediction));


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
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < inputs.length; j++) {
                double[] output = forward(inputs[j]);
                double[] error = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    error[k] = targets[j][k] - output[k];
                }
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
