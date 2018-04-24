import java.util.*;

public class Unit {

    private HashMap<Unit, Double> connectionWeights = new HashMap<>();
    private Double input;
    private Double output;
    private Double expectedOut;
    private Double error;
    private Double learningConstant;
    private Double momentum;
    private HashMap<Unit, Double> prvWeightChange = new HashMap<>();
    private HashMap<Unit, Double> netChanges = new HashMap<>();

    public Unit() {
        error = 0.0;
        learningConstant = 0.0;
        input = 0.0;
        output = 0.0;
        expectedOut = 0.0;
        momentum = 0.0;
    }

    public Unit (ArrayList<Unit> nextLayer, Double learningConstant, Double momentum) {
        Random rand = new Random();
        for (Unit i : nextLayer) {
            connectionWeights.put(i,rand.nextDouble() - 0.5);
            prvWeightChange.put(i, 0.0);
        }
        error = 0.0;
        expectedOut = 0.0;
        input = 0.0;
        output = 0.0;
        this.learningConstant = learningConstant;
        this.momentum = momentum;
        
    }
    
    public HashMap<Unit, Double> getConnections() {
        return connectionWeights;
    }

    public void setExpectedOut (Double out) {
        expectedOut = out;
    }
    
    public void reset (ArrayList<Unit> nextLayer) {
        connectionWeights = new HashMap<>();
        prvWeightChange = new HashMap<>();
        
        Random rand = new Random();
        for (Unit i : nextLayer) {
            connectionWeights.put(i,rand.nextDouble() - 0.5);
            prvWeightChange.put(i, 0.0);
        }
    }

    public void setInput (Double in) {
        input = in;
    }

    public void setOutput (Double out) {
        output = out;
        //System.out.println(out);
    }

    public void addInput (Double in) {
        input += in;
    }
    
    public Double getInput() {
        return input;
    }
    
    public Double getOutput() {
        return output;
    }

    public void forwardPropergate () {
        for (Unit i : connectionWeights.keySet()) {
            //System.out.println("Weight " + i + ": " + connectionWeights.get(i));
            i.addInput(connectionWeights.get(i)*output);
            //System.out.println("In: " + getInput());
        }
    }

    public void activation () {
        output = 1.0/(1.0 + Math.exp(-1*input));
    }
    
    public void outputActivation () {
        if(input >= 0.5) {
            output = 1.0;            
        } else {
            output = 0.0;
        }
    }
    
    public void resetError() {
        error = 0.0;
    }

    public void setError () {
        //System.out.println("In " + getInput());
        error = (expectedOut - output) * output * (1 - output);
        //System.out.println("Error " + error);
    }

    public void setErrorHidden () {
        Double errorSum = 0.0;
        for(Unit i : connectionWeights.keySet()) {
            errorSum += i.getError() * connectionWeights.get(i);
        }
        error = output*(1-output)*errorSum;
    }

    public Double getError () {
        return error;
    }
    
    public Double getExpectedOut() {
        return expectedOut;
    }
    
    public void resetNetChange() {
        netChanges = new HashMap<>();
    }
    
    public void adjustWeights () {
        for(Unit i : connectionWeights.keySet()) {
            connectionWeights.replace(i, connectionWeights.get(i) + 
                    (netChanges.get(i) + (momentum * prvWeightChange.get(i))));
            //System.out.println("Change: " + netChanges.get(i));
        }
    }
    
    public void setprvWeightChange () {
        for(Unit i : connectionWeights.keySet()) {
            prvWeightChange.replace(i, netChanges.get(i) + 
                    (momentum * prvWeightChange.get(i)));
        }
    }

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
}

    
