import java.util.*;

/**
 * Class representing a unit in a neural network
 * @author Daniel Thomson, ID:5040702, 2018
 */
public class Unit {

    //Hash map of connections to units and thier weights
    private HashMap<Unit, Double> connectionWeights = new HashMap<>();
    private Double input;
    private Double output;
    private Double expectedOut;
    private Double error;
    private Double learningConstant;
    private Double momentum;
    //Hash map of connections to units and the previous weight change from last
    //epoch
    private HashMap<Unit, Double> prvWeightChange = new HashMap<>();
    //Hash map of conenctions to units and the sum of weight change from each
    //pattern in current epoch.
    private HashMap<Unit, Double> netChanges = new HashMap<>();

    /**
     * Constructor for output units
     */
    public Unit() {
        error = 0.0;
        learningConstant = 0.0;
        input = 0.0;
        output = 0.0;
        expectedOut = 0.0;
        momentum = 0.0;
    }
    
    /**
     * Constructor for bias unit
     * 
     * @param hiddenLayer array of all units in hidden layer
     * @param outputLayer array of all units in output layer
     * @param momentum value of momentum as defined in parameter file
     * @param learningConstant value of the learning constant as defined in
     * parameter file.
     */
    public Unit(ArrayList<Unit> hiddenLayer, ArrayList<Unit> outputLayer, Double
             momentum, Double learningConstant) {
        error = 0.0;
        this.learningConstant = learningConstant;
        this.momentum = momentum;
        input = 0.0;
        output = 1.0;
        expectedOut = 0.0;
        Random rand = new Random();
        for (Unit i : hiddenLayer) {
            connectionWeights.put(i,(rand.nextDouble() - 0.5) * 0.6);
            prvWeightChange.put(i, 0.0);
        }
        for (Unit i : outputLayer) {
            connectionWeights.put(i,(rand.nextDouble() - 0.5) * 0.6);
            prvWeightChange.put(i, 0.0);
        }
    }

    /**
     * Constructor for hidden and input units
     * 
     * @param nextLayer array of units in next layer
     * @param learningConstant value of momentum as defined in parameter file
     * @param momentum value of the learning constant as defined in parameter
     * file
     */
    public Unit (ArrayList<Unit> nextLayer, Double learningConstant, Double momentum) {
        Random rand = new Random();
        for (Unit i : nextLayer) {
            connectionWeights.put(i,(rand.nextDouble() - 0.5) * 0.6);
            prvWeightChange.put(i, 0.0);
        }
        error = 0.0;
        expectedOut = 0.0;
        input = 0.0;
        output = 0.0;
        this.learningConstant = learningConstant;
        this.momentum = momentum;
        
    }
    
    /**
     * Returns a copy of connection weights, used in storing initial weight for
     * network reset
     */
    public HashMap<Unit, Double> copyWeights () {
        HashMap<Unit, Double> copy = new HashMap<>();
        for(Unit i : connectionWeights.keySet()) {
            copy.put(i, connectionWeights.get(i));
        }
        return copy;
    }
    
    /**
     * Re initializes the unit with new random value for its weight connections
     * 
     * @param nextLayer array of all unit in next layer
     */
    public void reset (ArrayList<Unit> nextLayer) {
        connectionWeights = new HashMap<>();
        prvWeightChange = new HashMap<>();
        
        Random rand = new Random();
        for (Unit i : nextLayer) {
            connectionWeights.put(i,(rand.nextDouble() - 0.5) * 0.6);
            prvWeightChange.put(i, 0.0);
        }
    }

    /**
     * Add some input to the units sum of inputs
     * 
     * @param in  
     */
    public void addInput (Double in) {
        input += in;
    }
    
    /**
     * Propagates the result of this units activation to connected units.
     */
    public void forwardPropergate () {
        for (Unit i : connectionWeights.keySet()) {
            //System.out.println("Weight " + i + ": " + connectionWeights.get(i));
            i.addInput(connectionWeights.get(i)*output);
            //System.out.println("In: " + getInput());
        }
    }

    /**
     * Pass the sum of inputs into this unit through its activation function, to
     * determine the units output.
     * Activation function: 1/(1 + e ^ -(sum of inputs))
     */
    public void activation () {
        output = 1.0/(1.0 + Math.exp(-1*input));
    }
    
    /**
     * Set the error this unit to 0
     */
    public void resetError() {
        error = 0.0;
    }

    /**
     * Set error of this unit. Only used on output units.
     */
    public void setError () {
        error = (expectedOut - output) * output * (1 - output);
    }

    /**
     * Set error of this unit. Only used on hidden units.
     */
    public void setErrorHidden () {
        Double errorSum = 0.0;
        for(Unit i : connectionWeights.keySet()) {
            errorSum += i.getError() * connectionWeights.get(i);
        }
        error = output*(1-output)*errorSum;
    }
    
    /**
     * Reset the hash map of net changes over epoch for each weight connection
     */
    public void resetNetChange() {
        netChanges = new HashMap<>();
    }
    
    /**
     * Adjust the weight value of this units connections based on the average
     * net change and portion of previous change (portion based on momentum).
     */
    public void adjustWeights () {
        for(Unit i : connectionWeights.keySet()) {
            connectionWeights.replace(i, connectionWeights.get(i) + 
                    (netChanges.get(i) + (momentum * prvWeightChange.get(i))));
        }
    }
    
    /**
     * Average out the weight change from each pattern over the epoch
     * 
     * @param inputCount number of patterns in this epoch
     */
    public void averageChange (int inputCount) {
        for(Unit i : netChanges.keySet()) {
            netChanges.replace(i, netChanges.get(i)/inputCount);
        }
    }
    
    /**
     * Stores the value of the previous weight changes in a hash map
     */
    public void setprvWeightChange () {
        for(Unit i : connectionWeights.keySet()) {
            prvWeightChange.replace(i, netChanges.get(i) + 
                    (momentum * prvWeightChange.get(i)));
        }
    }

    /**
     * Calculate the weight change for this pattern and add it to the sum of
     * change for the epoch.
     */
    public void addWeightChange () {
        for(Unit i : connectionWeights.keySet()) {
            Double change = learningConstant * i.getError() * output;
            if(netChanges.containsKey(i)) {
                Double newChange = netChanges.get(i) + change;
                netChanges.replace(i, newChange);
            } else {
                netChanges.put(i, change);
            }
        }
    }
    
    //Setters
    public void setInput (Double in) {
        input = in;
    }

    public void setOutput (Double out) {
        output = out;
    }

    public void setExpectedOut (Double out) {
        expectedOut = out;
    }
    
    public void setLearningConstant(Double lc) {
        learningConstant = lc;
    }
    
    public void setMomentum(Double mom) {
        momentum = mom;
    }
    
    public void setConnectionWeights(HashMap<Unit, Double> cw) {
        connectionWeights = cw;
    }
    
    //Getters
    public HashMap<Unit, Double> getConnections() {
        return connectionWeights;
    }
    
    public Double getInput() {
        return input;
    }
    
    public Double getOutput() {
        return output;
    }
    
    public Double getError () {
        return error;
    }
    
    public Double getExpectedOut() {
        return expectedOut;
    }
 
    public Double getLearningConstant() {
        return learningConstant;
    }
    
    public Double getMomentum() {
        return momentum;
    }

}

    
