package examples_yy;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.MultiSamplingEvaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.ClassifierLists;
import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Examples to show different ways of evaluating classifiers
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class Ex03_yy_BasicEvaluation {

    public static void main(String[] args) throws Exception {

        // We'll use this data throughout, see Ex01_Datahandling
        int seed = 0;
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances train = trainTest[0];
        Instances test = trainTest[1];

        // Let's use Random Forest throughout
        Classifier classifier = ClassifierLists.setClassifierClassic("RandF", seed);
        classifier.buildClassifier(train);

        // 使用SingleTestSetEvaluator
        boolean cloneData = true, setClassMissing = true;
        Evaluator testSetEval = new SingleTestSetEvaluator(seed, cloneData, setClassMissing);
        ClassifierResults testResults = testSetEval.evaluate(classifier, test);
        System.out.println("Random Forest accuracy on ItalyPowerDemand: " + testResults.getAcc());

        // 使用CrossValidationEvaluator
        boolean cloneClassifier = false, maintainFoldClassifiers = false;
        MultiSamplingEvaluator cvEval = new CrossValidationEvaluator(seed, cloneData, setClassMissing, cloneClassifier, maintainFoldClassifiers);
        cvEval.setNumFolds(10);

        ClassifierResults trainResults = cvEval.evaluate(classifier, train);
        System.out.println("Random Forest average accuracy estimate on ItalyPowerDemand: " + trainResults.getAcc());

        for (int i = 0; i < 10; i++)
            System.out.println("\tCVFold " + i + " accuracy: " + cvEval.getFoldResults()[i].getAcc());


        // 保存结果
        String mockFile = trainResults.writeFullResultsToString();
        System.out.println("\n\n" + mockFile);
    }

}
