import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.ejml.simple.SimpleMatrix;
import javax.swing.*;
import java.io.File;
import java.util.*;

public class main_code {
   public static void main(String[] args) {
      String path;
      ImageToColumnMatrix i = new ImageToColumnMatrix();
      nodeSet n = new nodeSet();
      imageLoader l = new imageLoader();
      int inputSize = 784;
      int hiddenSize1 = 16;
      int hiddenSize2 = 8;
      int outputSize = 3;

      SimpleMatrix W1;
      SimpleMatrix W2;
      SimpleMatrix W3, b1, b2, b3;
      Random random = new Random();

      // Initialize weight matrices with He initialization
      W1 = randomMatrix(hiddenSize1, inputSize);
      W2 = randomMatrix(hiddenSize2, hiddenSize1);
      W3 = randomMatrix(outputSize, hiddenSize2);
      // Initialize bias matrices with zeros
      b1 = new SimpleMatrix(hiddenSize1, 1);
      b2 = new SimpleMatrix(hiddenSize2, 1);
      b3 = new SimpleMatrix(outputSize, 1);

      while (true) {
         System.out.println("1. Load Image to test");
         System.out.println("2. Train model");
         System.out.println("3. Exit");
         System.out.println("Enter your choice: ");
         Scanner sc = new Scanner(System.in);
         int choice = sc.nextInt();
         switch (choice) {
            case 1:
               path = "C:\\Users\\Mathew\\Downloads\\archive (1)\\shapes\\circles\\drawing(1).png";
        
               System.out.println("Selected file: " + path);
               SimpleMatrix matrix1 = i.Matrices(path);
               if (matrix1 == null) {
                  System.err.println("Error: Could not convert image to matrix.");
                  break;
               }
               SimpleMatrix Matrix4 = n.firstnode(matrix1, W1, b1);
               SimpleMatrix Matrix5 = n.SecondNode(Matrix4, W2, b2);
               SimpleMatrix Output1 = n.outputNode(Matrix5, W3, b3);
               if (Output1.get(0, 0) > Output1.get(1, 0) && Output1.get(0, 0) > Output1.get(2, 0)) {
                  System.out.println("This image is a Triangle");
               } else if (Output1.get(1, 0) > Output1.get(0, 0) && Output1.get(1, 0) > Output1.get(2, 0)) {
                  System.out.println("This image is a square");
               } else {
                  System.out.println("The image is of circle");
               }
               break;
            case 2:
               backpropogation bp = new backpropogation();
               for (int i1 = 0; i1 < 300; i1++) {
                  SimpleMatrix Label = new SimpleMatrix(3, 1);
                  SimpleMatrix image_Matrix;
                  if (i1 < 100) {
                     Label.set(0, 0, 1); // Triangle
                     image_Matrix = l.loadImages("C:\\Users\\Mathew\\Downloads\\archive (1)\\shapes\\triangles", i1);
                  } else if (i1 >= 100 && i1 < 200) {
                     Label.set(1, 0, 1); // Square
                     image_Matrix = l.loadImages("C:\\Users\\Mathew\\Downloads\\archive (1)\\shapes\\squares", i1 - 100);
                  } else {
                     Label.set(2, 0, 1); // Circle
                     image_Matrix = l.loadImages("C:\\Users\\Mathew\\Downloads\\archive (1)\\shapes\\circles", i1 - 200);
                  }
                  if (image_Matrix == null) {
                     System.err.println("Error: Could not load image matrix for index " + i1);
                     continue;
                  }
                
                  if (image_Matrix.numRows() != inputSize || image_Matrix.numCols() != 1) {
                     System.err.println("Error: Image matrix dimensions do not match input size for index " + i1);
                     continue;
                  }
                  System.out.println("Training on image index: " + i1);
                  bp.train(image_Matrix, Label, W1, b1, W2, b2, W3, b3);
               }
               System.out.println("Model trained");
               break;
            case 3:
               System.exit(0);
               break;
            default:
               System.out.println("Invalid choice");
               break;
         }
      }
   }

   private static SimpleMatrix randomMatrix(int rows, int cols) {
      Random random = new Random();
      return SimpleMatrix.random_DDRM(rows, cols, -1, 1, random).scale(Math.sqrt(2.0 / cols));
   }

   public static String chooseFile() {
       JFileChooser fileChooser = new JFileChooser();
       fileChooser.setDialogTitle("Select a File");
       int result = fileChooser.showOpenDialog(null);
       
       if (result == JFileChooser.APPROVE_OPTION) {
           File selectedFile = fileChooser.getSelectedFile();
           return selectedFile.getAbsolutePath();
       }
       return null;
   }
}