import java.util.*;
import java.lang.Math.*;

public class Unit {

    private HashMap<Unit, Double> connectionWeights = new HashMap<>();
    private Double input;
    private Double output;
    private Double expectedOut;
    private Double error;
    private Double learningConstant;

    public Unit() {
        error = 0.0;
        learningConstant = 0.0;
        input = 0.0;
        output = 0.0;
        expectedOut = 0.0;
    }

    public Unit (ArrayList<Unit> nextLayer, Double learningConstant) {
        for (Unit i : nextLayer) {
            connectionWeights.put(i,0.0);
        }
        error = 0.0;
        expectedOut = 0.0;
        input = 0.0;
        output = 0.0;
        this.learningConstant = learningConstant;
    }

    public void setExpectedOut (Double out) {
        expectedOut = out;
    }

    public void setInput (Double in) {
        input = in;
    }

    public void setOutput (Double out) {
        output = out;
    }

    public void addInput (Double in) {
        input += in;
    }

    public void forwardPropergate () {
        for (Unit i : connectionWeights.keySet()) {
            i.addInput(connectionWeights.get(i)*output);
        }
    }

    public void activation () {
        output = 1.0/(1.0 + Math.exp(-1*input));
    }

    public void setError () {
        error = (expectedOut - output) * output * (1 - output);
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

    public void adjustWeights () {
        for(Unit i : connectionWeights.keySet()) {
            Double change = learningConstant * i.getError() * output;
            Double newWeight = connectionWeights.get(i) + change;
            connectionWeights.replace(i, newWeight);
        }
    }
}

    
