import org.ejml.simple.SimpleMatrix;

public class backpropogation {
    public void train(SimpleMatrix X, SimpleMatrix Y, SimpleMatrix W1, SimpleMatrix b1, SimpleMatrix W2, SimpleMatrix b2, SimpleMatrix W3, SimpleMatrix b3) {
        double learningRate = 0.1;
        Activations a = new Activations();
        // Forward pass
        SimpleMatrix z1 = W1.mult(X).plus(b1);
        SimpleMatrix a1 = a.relu(z1);

        SimpleMatrix z2 = W2.mult(a1).plus(b2);
        SimpleMatrix a2 = a.relu(z2);
        
        SimpleMatrix z3 = W3.mult(a2).plus(b3);
        SimpleMatrix a3 = a.softmax(z3);

        // Compute gradients (Loss derivative w.r.t output layer)
        SimpleMatrix dL_dz3 = a3.minus(Y);
        SimpleMatrix dL_dW3 = dL_dz3.mult(a2.transpose());
        SimpleMatrix dL_db3 = dL_dz3;

        // Backpropagate to second hidden layer
        SimpleMatrix dL_da2 = W3.transpose().mult(dL_dz3);
        SimpleMatrix dL_dz2 = dL_da2.elementMult(a.reluDerivative(z2));
        SimpleMatrix dL_dW2 = dL_dz2.mult(a1.transpose());
        SimpleMatrix dL_db2 = dL_dz2;

        // Backpropagate to first hidden layer
        SimpleMatrix dL_da1 = W2.transpose().mult(dL_dz2);
        SimpleMatrix dL_dz1 = dL_da1.elementMult(a.reluDerivative(z1));
        SimpleMatrix dL_dW1 = dL_dz1.mult(X.transpose());
        SimpleMatrix dL_db1 = dL_dz1;

        // Update weights and biases using gradient descent
        W3 = W3.minus(dL_dW3.scale(learningRate));
        b3 = b3.minus(dL_db3.scale(learningRate));

        W2 = W2.minus(dL_dW2.scale(learningRate));
        b2 = b2.minus(dL_db2.scale(learningRate));

        W1 = W1.minus(dL_dW1.scale(learningRate));
        b1 = b1.minus(dL_db1.scale(learningRate));
    }
}
