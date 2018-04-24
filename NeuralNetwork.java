import java.util.*;
import java.io.*;
import java.text.*;

public class NeuralNetwork {

    private static ArrayList<Unit> input = new ArrayList<>();
    private static ArrayList<Unit> hidden = new ArrayList<>();
    private static ArrayList<Unit> output = new ArrayList<>();
    private static ArrayList<String> pFileContent = new ArrayList<>();
    private static ArrayList<String> tFileContent = new ArrayList<>();
    private static ArrayList<String> iFileContent = new ArrayList<>();
    private static int epoch = 0;
    private static Double errorCriterion = 0.0;

    public static void main(String[]args) {
        Scanner sc = new Scanner(System.in);
        int command = 1;
        while (command != 0) {
            System.out.println("\nCOSC420 Assignment 1\n\nMenu:\n1 initialise\n2 learn"
                    + " (specify epochs)\n3 learn (to crietion)\n4 test\n5 show weights"
                    + "\n6 reset\n0 quit\n");
            command = sc.nextInt();
            switch(command) {
                case 1: 
                    readInput();
                    break;      
                case 2: 
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        learn(0);
                    }
                    break;
                case 3:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        learn(1);
                    }
                    break;
                case 4:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        test();
                    }
                    break;
                case 5:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        displayWeights();
                    }
                    break;
                case 6:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        reset();
                    }
                    break;
                case 0:
                    break;
                default:
                    System.out.println("Not valid input");
            }
        }
    }
    
    public static void reset () {
        for(Unit i : input) {
            i.reset(hidden);
        }
        for(Unit i : hidden) {
            i.reset(output);
        }
    }
    
    public static void displayWeights () {
        NumberFormat f = new DecimalFormat("#0.0000000");
        
        System.out.println("\nInput to Hidden Weights\n");
        for(int i = 0; i < hidden.size(); i++) {
            System.out.print("\t   Hidden" + (i+1));
        }
        for(int i = 0; i < input.size(); i++) {
            HashMap<Unit, Double> weights = input.get(i).getConnections();
            System.out.print("\nInput" + (i+1));
            for(int j = 0; j < hidden.size(); j++) {
                System.out.print("\t" + f.format(weights.get(hidden.get(j))));
            }
        }
        
        System.out.println("\n\nHidden to Output Weights\n");
        for(int i = 0; i < output.size(); i++) {
            System.out.print("\t   Output" + (i+1));
        }
        for(int i = 0; i < hidden.size(); i++) {
            HashMap<Unit, Double> weights = hidden.get(i).getConnections();
            System.out.print("\nHidden" + (i+1));
            for(int j = 0; j < output.size(); j++) {
                System.out.print("\t" + f.format(weights.get(output.get(j))));
            }
        }
        System.out.println("");
    }
    
    public static void test () {
        for(int i = 0; i < iFileContent.size(); i++) {
            System.out.println("Input: " + iFileContent.get(i));
            System.out.println("Expected Output: "+tFileContent.get(i)+"\n");
            setCycle(iFileContent.get(i), tFileContent.get(i));
            testCycle();
        }
    }
    
    public static void testCycle () {
        for(int i = 0; i < input.size(); i++) {
            System.out.println("Input" + i + ": " + input.get(i).getOutput());
            input.get(i).forwardPropergate();
        }
        for(int i = 0; i < hidden.size(); i++) {
            hidden.get(i).activation();
            System.out.println("Hidden" + i + ": " + hidden.get(i).getOutput());
            hidden.get(i).forwardPropergate();
        }
        for(int i = 0; i < output.size(); i++) {
            output.get(i).activation();
            if(output.get(i).getOutput() > 0.5) {
                System.out.println("Output" + i + ": 1.0");
            } else {
                System.out.println("Output" + i + ": 0.0");
            }
        }
        System.out.println("");
    }
    
    public static void learn (int stopType) {
        epoch = 0;
        Scanner sc = new Scanner(System.in);
        int epochLimit;
        Double currErrorCrit;
        if (stopType == 0) {
            System.out.println("Enter the amount of epochs");
            epochLimit = sc.nextInt();
            currErrorCrit = 0.0;
        } else {
            epochLimit = 10000;
            currErrorCrit = errorCriterion;
        }
        
        Double popError = 1.0;
        Double patternError = 0.0;
        while (popError > currErrorCrit && epoch < epochLimit) {
            popError = 0.0;
            patternError = 0.0;
            for(int i = 0; i < iFileContent.size(); i++) {
                setCycle(iFileContent.get(i), tFileContent.get(i));
                cycle();
                for(Unit ix : output) {
                    patternError += Math.pow(ix.getError(), 2);
                }
                //System.out.println("Pattern error: " + patternError);
                popError += patternError * 0.5;
            }
            popError /= Double.valueOf(pFileContent.get(2))*iFileContent.size();
            epoch++;
            if(epoch % 100 == 0) {
                System.out.println("Epoch " + epoch);
                System.out.println("Population Error: " + popError);
            }
            for (Unit i : hidden) {
                i.adjustWeights();
                i.setprvWeightChange();
                i.resetNetChange();
            }
            for(Unit i : input) {
                i.adjustWeights();
                i.setprvWeightChange();
                i.resetNetChange();
            }
        }
        System.out.println("Epoch " + epoch);
        System.out.println("Population Error: " + popError);
    }
    
    public static void readInput () {
        Scanner sc = new Scanner(System.in);
        String line = null;
        
        boolean validPFile = false;
        while(!validPFile) {
            System.out.println("Parameter file name");
            String pFile = sc.nextLine();
            pFileContent = new ArrayList<>();
            try {
                validPFile = true;
                BufferedReader br = new BufferedReader(new FileReader(pFile));
                while((line = br.readLine()) != null) {
                    pFileContent.add(line);
                }
            } catch(FileNotFoundException ex) {
                validPFile = false;
                System.out.println(pFile + " not found");
            } catch (IOException ex) {
                validPFile = false;
                ex.printStackTrace();
            }
        }
        output = new ArrayList<>();
        hidden = new ArrayList<>();
        input = new ArrayList<>();
        for(int i = 0; i < Integer.parseInt(pFileContent.get(2)); i++) {
            output.add(new Unit());
        }
        for(int i = 0; i < Integer.parseInt(pFileContent.get(1)); i++) {
            hidden.add(new Unit(output, Double.valueOf(pFileContent.get(3)), 
                    Double.valueOf(pFileContent.get(4))));
        }
        for(int i = 0; i < Integer.parseInt(pFileContent.get(0)); i++) {
            input.add(new Unit(hidden, Double.valueOf(pFileContent.get(3)), 
                    Double.valueOf(pFileContent.get(4))));
        }
        errorCriterion = Double.valueOf(pFileContent.get(5));
        
        boolean validIFile = false;
        while(!validIFile) {
            System.out.println("Input file name");
            String iFile = sc.nextLine();
            iFileContent = new ArrayList<>();
            try {
                validIFile = true;
                BufferedReader br = new BufferedReader(new FileReader(iFile));
                while((line = br.readLine()) != null) {
                    iFileContent.add(line);
                }
            } catch (FileNotFoundException ex) {
                validIFile = false;
                System.out.println(iFile + " not found");
            } catch (IOException ex) {
                validIFile = false;
                ex.printStackTrace();
            }
        }

        boolean validTFile = false;
        while(!validTFile) {
            System.out.println("Teacher file name");
            String tFile = sc.nextLine();
            tFileContent = new ArrayList<>();
            try {
                validTFile = true;
                BufferedReader br = new BufferedReader(new FileReader(tFile));
                while((line = br.readLine()) != null) {
                    tFileContent.add(line);
                }
            } catch (FileNotFoundException ex) {
                validTFile = false;
                System.out.println(tFile + "not found");
            } catch (IOException ex) {
                validTFile = false;
                ex.printStackTrace();
            }
        }
    }

    public static void setCycle(String inputString, String expectedOutput) {
        String[] splitInput = inputString.split(" +");
        for(int i = 0; i < splitInput.length; i++) {
            //System.out.print("Input " + i + " ");
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
            //System.out.println("Input " + i);
            i.forwardPropergate();
        }
        for(Unit i : hidden) {
            //System.out.println("Hidden " + i);
            i.activation();
            i.forwardPropergate();
        }
        for(Unit i : output) {
            i.outputActivation();
            i.setError();
            //System.out.println("Output " + i.getOutput());
        }
        for(Unit i : hidden) {
            i.addWeightChange();
            i.setErrorHidden();
        }
        for(Unit i : input) {
            i.addWeightChange();
        }
    }
}
