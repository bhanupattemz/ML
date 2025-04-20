import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class experiment6 {
    public static void main(String[] args) throws Exception {
        // Load dataset from ARFF or CSV file
        DataSource source = new DataSource("documents.arff"); // use .csv if you want
        Instances data = source.getDataSet();

        // Set class index to last attribute
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Create and train Naive Bayes model
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);

        // Evaluate model with 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(nb, data, 10, new Random(1));

        // Output metrics
        System.out.println("Accuracy: " + ((1 - eval.errorRate()) * 100) + "%");
        System.out.println("Precision: " + eval.weightedPrecision());
        System.out.println("Recall: " + eval.weightedRecall());
    }
}
