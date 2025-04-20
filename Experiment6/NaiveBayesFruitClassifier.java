import smile.classification.NaiveBayes;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.validation.CrossValidation;
import smile.validation.Accuracy;

import java.io.IOException;

public class NaiveBayesFruitClassifier {
    public static void main(String[] args) throws IOException {
        // Load the dataset (CSV or ARFF file)
        DataFrame data = Read.csv("c:\\\\Users\\\\bhanu\\\\OneDrive\\\\Desktop\\\\@jntua\\\\ML_lab\\\\Prac\\\\fruit.csv");

        // Use Smile's Cross-validation to evaluate accuracy
        Formula formula = Formula.lhs("target");  // Assuming "target" is the column to predict
        NaiveBayes model = NaiveBayes.fit(formula, data); // Fit the model on the training data

        // Evaluate model performance
        double accuracy = CrossValidation.test(model, data, 5);
        System.out.println("Accuracy: " + accuracy * 100 + "%");
    }
}
