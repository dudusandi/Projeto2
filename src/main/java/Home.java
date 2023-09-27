import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Home {

    public static void main(String[] args)throws IOException {
        Reader reader = Files.newBufferedReader(Paths.get("data.csv"));
        CSVParser parser = new CSVParserBuilder().withSeparator(';').build();
        CSVReader csvReader = new CSVReaderBuilder(reader)
                .withCSVParser(parser)
                .withSkipLines(1)
                .build();
        List<String[]> Home = csvReader.readAll();
        for (String[] home: Home){
            System.out.println(home[1]);
        }




    }
}
