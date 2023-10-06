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

    private int pesoEntrada;
    private int pesoOculto;
    private int pesoSaida;
    private double[][] entradaPesoOculto;
    private double[][] pesoOcultoSaida;
    private double[] pesosOcultos;
    private double[] saidasOcultas;
    private Random random;

    public static void main(String[] args) {


        // Leitura do CSV
        try {
            Reader reader = Files.newBufferedReader(Paths.get("data.csv"));
            CSVParser parser = new CSVParserBuilder().withSeparator(',').build();
            CSVReader csvReader = new CSVReaderBuilder(reader)
                    .withCSVParser(parser)
                    .withSkipLines(1)
                    .build();

            List<String[]> data = csvReader.readAll();

            //Criaçao dos Arrays
            List<double[]> entradas = new ArrayList<>();
            List<double[]> saidas = new ArrayList<>();

            for (String[] row : data) {
                double[] input = new double[13]; // Numero de Entrada de Dados
                for (int i = 2; i < row.length; i++) {
                    input[i - 2] = Double.parseDouble(row[i]);
                }
                entradas.add(input);

                double[] saida = new double[3]; // Definir outputSize com o número correto de saídas

                // Mapear "H", "A" e "D" para valores numéricos
                String result = row[0];
                if ("H".equals(result)) {
                    saida[0] = 0;
                    saida[1] = 1;
                    saida[2] = 0;
                } else if ("A".equals(result)) {
                    saida[0] = 1;
                    saida[1] = 0;
                    saida[2] = 0;
                } else if ("D".equals(result)) {
                    saida[0] = 0;
                    saida[1] = 0;
                    saida[2] = 1;
                }

                saidas.add(saida);
            }

            //Configuraçao da Previsão
            int tamanhoEntrada = 13;      // Tamanho  de entrada
            int cadamadaOculta = 10;      // Tamanho da camada oculta
            int tamanhoSaida = 3;         // Tamanho de saída
            int geracoes = 1000;          // Número de gerações de treinamento
            double taxaAprendizado = 0.7; // Taxa de aprendizado

            // Treinar a rede neural
            Home home = new Home(tamanhoEntrada, cadamadaOculta, tamanhoSaida);
            home.treinar(entradas.toArray(new double[0][0]), saidas.toArray(new double[0][0]), geracoes, taxaAprendizado);

            //Listar Entradas e fazer a Previsão
            List<double[]> previsoes = new ArrayList<>();

            for (double[] inputToPredict : entradas) {
                double[] previsao = home.calculaSaida(inputToPredict);
                previsoes.add(previsao);
            }

            System.out.println("Foram Analisadas " + entradas.size() + " entradas");
            double[] finalResult = calcularMedia(previsoes);
            System.out.println("Resultado final: " + Arrays.toString(finalResult));

        } catch (IOException e) {
            System.out.println(e.getMessage());
        }


    }


    public Home(int pesoEntrada, int pesoOculto, int pesoSaida) {
        this.pesoEntrada = pesoEntrada;
        this.pesoOculto = pesoOculto;
        this.pesoSaida = pesoSaida;
        entradaPesoOculto = new double[pesoEntrada][pesoOculto];
        pesoOcultoSaida = new double[pesoOculto][pesoSaida];
        pesosOcultos = new double[pesoOculto];
        saidasOcultas = new double[pesoSaida];
        random = new Random();
        peso();
    }

    // Calcula a media de todas as Linhas
    private static double[] calcularMedia(List<double[]> previsoes) {
        if (previsoes.isEmpty()) {
            return null;
        }
        int outputSize = previsoes.get(0).length;
        double[] average = new double[outputSize];
        for (double[] previsao : previsoes) {
            for (int i = 0; i < outputSize; i++) {
                average[i] += previsao[i];
            }
        }
        for (int i = 0; i < outputSize; i++) {
            average[i] /= previsoes.size();
        }
        return average;
    }


    //Calcula as saídas da rede neural
    public double[] calculaSaida(double[] input) {
        double[] hidden = new double[pesoOculto];
        for (int i = 0; i < pesoOculto; i++) {
            hidden[i] = pesosOcultos[i];
            for (int j = 0; j < pesoEntrada; j++) {
                hidden[i] += input[j] * entradaPesoOculto[j][i];
            }
            hidden[i] = sigmoid(hidden[i]);
        }
        double[] output = new double[pesoSaida];
        for (int i = 0; i < pesoSaida; i++) {
            output[i] = saidasOcultas[i];
            for (int j = 0; j < pesoOculto; j++) {
                output[i] += hidden[j] * pesoOcultoSaida[j][i];
            }
            output[i] = sigmoid(output[i]);
        }
        return output;
    }


    // Treinamento
    public void treinar(double[][] inputs, double[][] saidas, int geracao, double taxaAprendizado) {
        for (int i = 0; i < geracao; i++) {
            for (int j = 0; j < inputs.length; j++) {
                double[] saida = calculaSaida(inputs[j]);
                double[] erro = new double[pesoSaida];
                for (int k = 0; k < pesoSaida; k++) {
                    erro[k] = saidas[j][k] - saida[k];
                }
                double[] erroOculto = new double[pesoOculto];
                for (int k = 0; k < pesoOculto; k++) {
                    erroOculto[k] = 0;
                    for (int l = 0; l < pesoSaida; l++) {
                        erroOculto[k] += erro[l] * pesoOcultoSaida[k][l];
                    }
                }
                double[] hidden = new double[pesoOculto];
                for (int k = 0; k < pesoSaida; k++) {
                    saidasOcultas[k] += taxaAprendizado * erro[k];
                    for (int l = 0; l < pesoOculto; l++) {
                        pesoOcultoSaida[l][k] += taxaAprendizado * erro[k] * hidden[l];
                    }
                }
                for (int k = 0; k < pesoOculto; k++) {
                    pesosOcultos[k] += taxaAprendizado * erroOculto[k];
                    for (int l = 0; l < pesoEntrada; l++) {
                        entradaPesoOculto[l][k] += taxaAprendizado * erroOculto[k] * inputs[j][l];
                    }
                }
            }
        }
    }


    // Peso
    private void peso() {
        for (int i = 0; i < pesoEntrada; i++) {
            for (int j = 0; j < pesoOculto; j++) {
                entradaPesoOculto[i][j] = random.nextGaussian();
            }
        }
        for (int i = 0; i < pesoOculto; i++) {
            for (int j = 0; j < pesoSaida; j++) {
                pesoOcultoSaida[i][j] = random.nextGaussian();
            }
        }
    }


    //Função de Aprendizado
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));


    }
}
