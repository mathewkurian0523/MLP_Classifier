import org.ejml.simple.SimpleMatrix;

public class nodeSet implements nodes {
    Activations a = new Activations();

    @Override
    public SimpleMatrix firstnode(SimpleMatrix input, SimpleMatrix weights, SimpleMatrix bias) {
        SimpleMatrix output1 = weights.mult(input).plus(bias);
        return a.relu(output1);
    }

    @Override
    public SimpleMatrix SecondNode(SimpleMatrix input2, SimpleMatrix weights2, SimpleMatrix bias2) {
        SimpleMatrix output1 = weights2.mult(input2).plus(bias2);
        return a.relu(output1);
    }

    @Override
    public SimpleMatrix outputNode(SimpleMatrix input3, SimpleMatrix weights3, SimpleMatrix bias3) {
        SimpleMatrix output1 = weights3.mult(input3).plus(bias3);
        return a.softmax(output1);
    }
}
