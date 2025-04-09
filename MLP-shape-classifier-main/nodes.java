import org.ejml.simple.SimpleMatrix;

public interface nodes {
    public SimpleMatrix firstnode(SimpleMatrix input, SimpleMatrix weights, SimpleMatrix bias);
    public SimpleMatrix SecondNode(SimpleMatrix input, SimpleMatrix weights, SimpleMatrix bias);
    public SimpleMatrix outputNode(SimpleMatrix input, SimpleMatrix weights, SimpleMatrix bias);
}
