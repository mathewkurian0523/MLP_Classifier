import javax.swing.*;
import java.io.File;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;
import java.util.*;

public class ShapeClassifier {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static String getUserInputPath() {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the path to the image file: ");
        return scanner.nextLine();
    }

    public static Mat convertToGrayscale(String filePath) {
        Mat image = Imgcodecs.imread(filePath, Imgcodecs.IMREAD_GRAYSCALE);
        Imgproc.resize(image, image, new Size(28, 28)); // Ensure image is 28x28 pixels
        return image;
    }

    public static ANN_MLP loadPretrainedModel(String modelPath) {
        ANN_MLP mlp = ANN_MLP.load(modelPath);
        return mlp;
    }

    public static String classifyImage(Mat image, ANN_MLP mlp) {
        Mat flattened = image.reshape(1, 1);
        flattened.convertTo(flattened, CvType.CV_32F);

        Mat output = new Mat();
        mlp.predict(flattened, output);
        
        Core.MinMaxLocResult result = Core.minMaxLoc(output);
        int classIndex = (int) result.maxLoc.x;

        switch (classIndex) {
            case 0: return "Circle";
            case 1: return "Square";
            case 2: return "Triangle";
            default: return "Unknown";
        }
    }

    public static void trainModel(String datasetPath, String modelPath) {
        File datasetDir = new File(datasetPath);
        File[] files = datasetDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));
        if (files == null) {
            System.out.println("No images found in dataset directory.");
            return;
        }

        List<Mat> trainingData = new ArrayList<>();
        List<Mat> labels = new ArrayList<>();

        for (File file : files) {
            Mat img = convertToGrayscale(file.getAbsolutePath());
            Mat flattened = img.reshape(1, 1);
            flattened.convertTo(flattened, CvType.CV_32F);
            trainingData.add(flattened);

            Mat label = new Mat(1, 3, CvType.CV_32F, new Scalar(0));
            if (file.getName().contains("circle")) label.put(0, 0, 1);
            else if (file.getName().contains("square")) label.put(0, 1, 1);
            else if (file.getName().contains("triangle")) label.put(0, 2, 1);
            labels.add(label);
        }

        Mat trainData = new Mat();
        Mat trainLabels = new Mat();
        Core.vconcat(trainingData, trainData);
        Core.vconcat(labels, trainLabels);

        ANN_MLP mlp = ANN_MLP.create();
        mlp.setLayerSizes(new MatOfInt(784, 128, 128, 128, 3));
        mlp.setTrainMethod(ANN_MLP.BACKPROP);
        mlp.setBackpropWeightScale(0.1);
        mlp.setBackpropMomentumScale(0.1);
        mlp.setActivationFunction(ANN_MLP.SIGMOID_SYM);

        mlp.train(trainData, Ml.ROW_SAMPLE, trainLabels);
        mlp.save(modelPath);
        System.out.println("Model trained and saved to " + modelPath);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter dataset path for training or press Enter to classify an image: ");
        String datasetPath = scanner.nextLine();
        String modelPath = "mlp_model.xml";

        if (!datasetPath.isEmpty()) {
            trainModel(datasetPath, modelPath);
        } else {
            ANN_MLP mlp = loadPretrainedModel(modelPath);
            String filePath = getUserInputPath();
            Mat image = convertToGrayscale(filePath);
            String shape = classifyImage(image, mlp);
            System.out.println("Classified as: " + shape);
        }
    }
}
