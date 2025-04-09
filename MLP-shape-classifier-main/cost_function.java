import org.ejml.simple.SimpleMatrix;

public class cost_function {
    public double cost(SimpleMatrix input, SimpleMatrix label){
        SimpleMatrix Cost = new SimpleMatrix(3,1);
        for (int i = 0; i < 3; i++) {
            double input1 = input.get(i, 0);
            double label1 = label.get(i, 0);
            double cost = -label1 * Math.log(input1);
            Cost.set(i, 0, cost);
        }
        double Cost_sum = Cost.elementSum();
        return Cost_sum;
    }
}