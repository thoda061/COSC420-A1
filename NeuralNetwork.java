import java.util.*;
import java.io.*;

public class NeuralNetwork {

    private static ArrayList<Unit> input = new ArrayList<>();
    private static ArrayList<Unit> hidden = new ArrayList<>();
    private static ArrayList<Unit> output = new ArrayList<>();
    private static int epoch = 0;
    private static Double errorCriterion;

    public static void main(String[]args) {
        Scanner sc = new Scanner(System.in);
        String line = null;
        
        System.out.println("Parameter file name");
        String pFile = sc.nextLine();
        ArrayList<String> pFileContent = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(pFile));
            while((line = br.readLine()) != null) {
                pFileContent.add(line);
            }
        } catch(FileNotFoundException ex) {
            System.out.println(pFile + " not found");
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        for(int i = 0; i < Integer.parseInt(pFileContent.get(2)); i++) {
            output.add(new Unit());
        }
        for(int i = 0; i < Integer.parseInt(pFileContent.get(1)); i++) {
            hidden.add(new Unit(output, Double.valueOf(pFileContent.get(3))));
        }
        for(int i = 0; i < Integer.parseInt(pFileContent.get(0)); i++) {
            input.add(new Unit(hidden, Double.valueOf(pFileContent.get(3))));
        }
        errorCriterion = Double.valueOf(pFileContent.get(5));
        
        System.out.println("Input file name");
        String iFile = sc.nextLine();
        ArrayList<String> iFileContent = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(iFile));
            while((line = br.readLine()) != null) {
                iFileContent.add(line);
            }
        } catch (FileNotFoundException ex) {
            System.out.println(iFile + " not found");
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        System.out.println("Teacher file name");
        String tFile = sc.nextLine();
        ArrayList<String> tFileContent = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(tFile));
            while((line = br.readLine()) != null) {
                tFileContent.add(line);
            }
        } catch (FileNotFoundException ex) {
            System.out.println(tFile + "not found");
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        while (epoch < 100) {
            for(int i = 0; i < iFileContent.size(); i++) {
                setCycle(iFileContent.get(i), tFileContent.get(i));
                cycle();
            }
            epoch++;
            System.out.println(epoch);
        }
    }

    public static void setCycle(String inputString, String expectedOutput) {
        String[] splitInput = inputString.split(" ");
        for(int i = 0; i < splitInput.length; i++) {
            input.get(i).setOutput(Double.valueOf(splitInput[i]));
        }
        String[] splitOutput = expectedOutput.split(" ");
        for(int i = 0; i < splitOutput.length; i++) {
            Unit x = output.get(i);
            x.setExpectedOut(Double.valueOf(splitOutput[i]));
            x.setInput(0.0);
        }
        for(Unit i : hidden) {
            i.setInput(0.0);
        }   
    }

    public static void cycle() {
        for(Unit i : input) {
            i.forwardPropergate();
        }
        for(Unit i : hidden) {
            i.activation();
            i.forwardPropergate();
        }
        for(Unit i : output) {
            i.activation();
            i.setError();
            System.out.println(i.getError());
        }
        for(Unit i : hidden) {
            i.setErrorHidden();
            i.adjustWeights();
        }
        for(Unit i : input) {
            i.adjustWeights();
        }
    }
}
