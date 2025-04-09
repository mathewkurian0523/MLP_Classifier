import java.io.File;
import org.ejml.simple.SimpleMatrix;

public class imageLoader {
    public SimpleMatrix loadImages(String directoryPath, int i) {
        ImageToColumnMatrix t = new ImageToColumnMatrix();
        File directory = new File(directoryPath);
        File[] files = directory.listFiles();
        if (files == null || i >= files.length) {
            System.err.println("Error: Invalid file index or directory is empty.");
            return null;
        }
        File CountFile = files[i];
        String filePath = CountFile.getAbsolutePath();
        return t.Matrices(filePath);
    }
}