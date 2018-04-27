import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Class for building and using a neural network
 * @author Daniel Thomson, ID:5040702, 2018
 */
public class NeuralNetwork {

    //Array of units in the input layer
    private static ArrayList<Unit> input = new ArrayList<>();
    //Array of units in the hidden layer
    private static ArrayList<Unit> hidden = new ArrayList<>();
    //Array of units in the output layer
    private static ArrayList<Unit> output = new ArrayList<>();
    //Bias unit
    private static Unit bias;
    //Content of the parameter file
    private static ArrayList<String> pFileContent = new ArrayList<>();
    //Content of the teacher file
    private static ArrayList<String> tFileContent = new ArrayList<>();
    //Content of the input file
    private static ArrayList<String> iFileContent = new ArrayList<>();
    private static int epoch = 0;
    private static Double errorCriterion = 0.0;
    private static ArrayList<HashMap<Unit, Double>> startWeights = new ArrayList<>();

    /**
     * Main program execution, runs the program's main menu until user enter
     * 0 to exit
     * @param args command line arguments (not used)
     */
    public static void main(String[]args) {
        Scanner sc = new Scanner(System.in);
        int command = 1;
        while (command != 0) {
            System.out.println("\nCOSC420 Assignment 1\n\nMenu:\n1 initialise"
                    + "\n2 change learning parameters\n3 learn\n4 test\n5 show weights"
                    + "\n6 show learning parameters\n7 reset\n0 quit\n");
            command = sc.nextInt();
            switch(command) {
                case 1: 
                    readInput();
                    break;
                case 2:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        System.out.println("\nChange Parameters\n1 change learning constant"
                                + "\n2 change momentum\n3 change error criterion\n0 back");
                        int subCommand = 1;
                        subCommand = sc.nextInt();
                        switch(subCommand) {
                            case 1:
                                updateParam(1);
                                break;
                            case 2:
                                updateParam(2);
                                break;
                            case 3:
                                updateParam(3);
                                break;
                            case 0:
                                break;
                            default:
                                System.out.println("Invalid command");
                                break;
                        }
                    }
                    break;
                case 3: 
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        System.out.println("\nLearn\n1 learn to criterion\n2"
                                + " learn to specified epoch\n0 back");
                        int subCommand = sc.nextInt();
                        switch(subCommand) {
                            case 1:
                                learn(1, iFileContent, tFileContent);
                                break;
                            case 2:
                                learn(0, iFileContent, tFileContent);
                                break;
                            case 0:
                                break;
                            default:
                                System.out.println("Invalid command");
                                break;
                        }     
                    }
                    break;
                case 4:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        int matchCount = test();
                        System.out.println("Number of correct outputs: "
                                + matchCount + "/" + iFileContent.size());
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
                        System.out.println("Learning Constant: " +
                                input.get(0).getLearningConstant());
                        System.out.println("Momentum: " + 
                                input.get(0).getMomentum());
                        System.out.println("Error Criterion: " + errorCriterion);
                    }
                    break;
                case 7:
                    if(errorCriterion == 0.0) {
                        System.out.println("Must initialise first");
                    } else {
                        System.out.println("\nReset\n1 reset with starting weights"
                                + "\n2 reset with new random weights\n0 back");
                        int subCommand = sc.nextInt();
                        switch(subCommand) {
                            case 1:
                                resetBack();
                                break;
                            case 2:
                                reset();
                                break;
                            case 0:
                                break;
                            default:
                                System.out.println("Invalid command");
                                break;
                        }
                    }
                    break;
                case 0:
                    break;
                default:
                    System.out.println("Not valid input");
            }
        }
    }
    
    /**
     * Ask for user to enter value with which to update learning parameters
     * @param param type of parameter to be updated, 1 for learning constant,
     * 2 for momentum and 3 for error criterion.
     */
    public static void updateParam(int param) {
        Scanner sc = new Scanner(System.in);
        if(param == 1) {
            System.out.println("Enter new learning constant");
            Double lc = sc.nextDouble();
            bias.setLearningConstant(lc);
            for(Unit i : hidden) {
                i.setLearningConstant(lc);
            }
            for(Unit i : input) {
                i.setLearningConstant(lc);
            }
        } else if (param == 2) {
            System.out.println("Enter new momentum");
            Double mom = sc.nextDouble();
            bias.setMomentum(mom);
            for(Unit i : hidden) {
                i.setMomentum(mom);
            }
            for(Unit i : input) {
                i.setMomentum(mom);
            }
        } else {
            System.out.println("Enter new error criterion");
            errorCriterion = sc.nextDouble();
        }
    }
    
    public static void resetBack () {
        for(int i = 0; i < input.size(); i++) {
            input.get(i).setConnectionWeights(startWeights.get(i));
        }
        for(int i = input.size(); i < input.size() + hidden.size(); i++) {
            hidden.get(i - input.size()).setConnectionWeights(startWeights.get(i));
        }
        bias.setConnectionWeights(startWeights.get(startWeights.size() - 1));
        setStartWeights();
    }
    
    /**
     * Reset the weight connections of all unit in the network with new random
     * weights
     */
    public static void reset () {
        for(Unit i : input) {
            i.reset(hidden);
        }
        for(Unit i : hidden) {
            i.reset(output);
        }
        ArrayList<Unit> biasConnect = new ArrayList<>();
        for(Unit i : hidden) {
            biasConnect.add(i);
        }
        for(Unit i : output) {
            biasConnect.add(i);
        }
        bias.reset(biasConnect);
        setStartWeights();
    }
    
    /**
     * Prints out all the unit connection in the network. Prints three tables :
     * Input to Hidden weights, Hidden to Output weights and bias weights.
     */
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
        
        System.out.println("\nBias Weights\n\n\tBias1");
        HashMap<Unit, Double> biasWeights = bias.getConnections();
        for(int i = 0; i < hidden.size(); i++) {
            System.out.println("Hidden" + (i+1) + "\t" + 
                    f.format(biasWeights.get(hidden.get(i))));
        }
        for(int i = 0; i < output.size(); i++) {
            System.out.println("Output" + (i+1) + "\t" + 
                    f.format(biasWeights.get(output.get(i))));
        }
        
    }
    
    /**
     * Tests the network by doing forward propergation and comparing network 
     * output to expected output. Show all unit activations during propergation.
     */
    public static int test () {
        int matchCount = 0;
        for(int i = 0; i < iFileContent.size(); i++) {
            System.out.println("Input: " + iFileContent.get(i));
            System.out.println("Expected Output: "+tFileContent.get(i)+"\n");
            setCycle(iFileContent.get(i), tFileContent.get(i));
            int[] outputVal = testCycle();
            matchCount += confirmMatch(outputVal, tFileContent.get(i));
        }
        return matchCount;
    }
    
    /**
     * Confirms network output matches expected output
     * @param outputVal network output
     * @param expectedVal expected output
     * @return 1 if match, 0 if not
     */
    public static int confirmMatch (int[] outputVal, String expectedVal) {
        boolean match = true;
        String [] split = expectedVal.split(" ");
        for(int i = 0; i < outputVal.length; i++) {
            if(outputVal[i] != Double.valueOf(split[i])) {
                match = false;
            }
        }
        if (match) {
            return 1;
        } else {
            return 0;
        }
    }
    
    /**
     * Performs forward propagation for a single pattern
     * @return output of propagation
     */
    public static int[] testCycle () {
        int [] outputVal = new int[output.size()]; 
        bias.forwardPropergate();
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
                outputVal[i] = 1;
            } else {
                System.out.println("Output" + i + ": 0.0");
                outputVal[i] = 0;
            }
        }
        System.out.println("");
        return outputVal;
    }
    
    /**
     * Repeatedly goes through the process of forward and back propagation in 
     * order to teach the network, until reached specified amount of epoch or 
     * population error is less than error criterion.
     * 
     * @param stopType determines whether to train to error criterion or to 
     * specified epoch
     * @param inFile Array list of input patterns to train with.
     * @param outFile Array list of expected output patterns to train with.
     */
    public static void learn (int stopType, ArrayList<String> inFile, 
            ArrayList<String> outFile) {
        epoch = 0;
        Scanner sc = new Scanner(System.in);
        int epochLimit;
        Double currErrorCrit;
        if (stopType == 0) {
            System.out.println("Enter the amount of epochs");
            epochLimit = sc.nextInt();
            currErrorCrit = 0.0;
        } else {
            epochLimit = 500000;
            currErrorCrit = errorCriterion;
        }
        
        Double popError = 1.0;
        Double patternError = 0.0;
        while (popError > currErrorCrit && epoch < epochLimit) {
            popError = 0.0;
            patternError = 0.0;
            for(int i = 0; i < inFile.size(); i++) {
                setCycle(inFile.get(i), outFile.get(i));
                cycle();
                for(Unit ix : output) {
                    patternError += Math.pow(ix.getExpectedOut() - 
                            ix.getOutput(), 2);
                }
                popError += patternError * 0.5;
            }
            popError /= Double.valueOf(pFileContent.get(2))*inFile.size();
            epoch++;
            if(epoch % 100 == 0) {
                System.out.println("Epoch " + epoch);
                System.out.println("Population Error: " + popError);
            }
            for(Unit i : hidden) {
                i.averageChange(inFile.size());
                i.adjustWeights();
                i.setprvWeightChange();
                i.resetNetChange();
            }
            for(Unit i : input) {
                i.averageChange(inFile.size());
                i.adjustWeights();
                i.setprvWeightChange();
                i.resetNetChange();
            }
            bias.averageChange(inFile.size());
            bias.adjustWeights();
            bias.setprvWeightChange();
            bias.resetNetChange();
        }
        System.out.println("Epoch " + epoch);
        System.out.println("Population Error: " + popError);
    }
    
    /**
     * Stores the all the connection weight value currently set in the network.
     * Used to reset the network back to initial weights
     */
    public static void setStartWeights () {
        startWeights = new ArrayList<>();
        for(Unit i : input) {
            startWeights.add(i.copyWeights());
        }
        for(Unit i : hidden) {
            startWeights.add(i.copyWeights());
        }
        startWeights.add(bias.copyWeights());
    }
    
    /**
     * Ask for and read three files, one with network parameter, one with input 
     * patterns and one with expected output patterns. Uses data in these files 
     * to initialize the network and train it. 
     */
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
        bias = new Unit(hidden, output, Double.valueOf(pFileContent.get(3)), 
                Double.valueOf(pFileContent.get(4)));
        setStartWeights();
        
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

    /**
     * Sets the input and expected output for a pattern
     * @param inputString
     * @param expectedOutput 
     */
    public static void setCycle(String inputString, String expectedOutput) {
        String[] splitInput = inputString.split(" +");
        for(int i = 0; i < splitInput.length; i++) {
            //System.out.print("Input " + i + " ");
            input.get(i).setOutput(Double.valueOf(splitInput[i]));
        }
        String[] splitOutput = expectedOutput.split(" +");
        for(int i = 0; i < splitOutput.length; i++) {
            Unit x = output.get(i);
            x.setExpectedOut(Double.valueOf(splitOutput[i]));
            x.setInput(0.0);
        }
        for(Unit i : hidden) {
            i.setInput(0.0);
        }   
    }

    /**
     * Performs forward and back propagation for set pattern.
     */
    public static void cycle() {
        bias.forwardPropergate();
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
        }
        for(Unit i : hidden) {
            i.addWeightChange();
            i.setErrorHidden();
        }
        for(Unit i : input) {
            i.addWeightChange();
        }
        bias.addWeightChange();
    }
}
