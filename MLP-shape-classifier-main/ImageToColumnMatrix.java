import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.ejml.simple.SimpleMatrix;

public class ImageToColumnMatrix {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); 
    }

    public SimpleMatrix Matrices(String imagePath) {
        SimpleMatrix nullMatrix = null;
        Mat image = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        if (image.empty()) {
            System.err.println("Error: Cannot load image.");
            return nullMatrix;
        }
        int rows = image.rows();
        int cols = image.cols();


        SimpleMatrix matrix = new SimpleMatrix(rows * cols, 1);

        int index = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double pixelValue = image.get(i, j)[0];
                matrix.set(index++, 0, pixelValue / 255.0); 
            }

        return matrix;
    }
}
