import org.ejml.simple.SimpleMatrix;

public class Activations {
 
    SimpleMatrix relu(SimpleMatrix matrix) {
        for (int i = 0; i < matrix.getNumElements(); i++) {
            matrix.set(i, Math.max(0, matrix.get(i)));
        }
        return matrix;
    }

    SimpleMatrix softmax(SimpleMatrix matrix){
        double max = matrix.elementMaxAbs();
        SimpleMatrix expMatrix = new SimpleMatrix(matrix.numRows(), matrix.numCols());
        double sum = 0.0;
        for (int i = 0; i < matrix.getNumRows(); i++) {
            double expValue = Math.exp(matrix.get(i, 0) - max);
            expMatrix.set(i, 0, expValue);
            sum += expValue;
        }
        for (int i = 0; i < matrix.getNumRows(); i++) {
            expMatrix.set(i, 0, expMatrix.get(i, 0) / sum);
        }
        return expMatrix;
    }

    SimpleMatrix reluDerivative(SimpleMatrix matrix) {
        for (int i = 0; i < matrix.getNumElements(); i++) {
            matrix.set(i, matrix.get(i) > 0 ? 1 : 0);
        }
        return matrix;
    }

}
