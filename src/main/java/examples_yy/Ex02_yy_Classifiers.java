package examples_yy;


import experiments.data.DatasetLoading;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class Ex02_yy_Classifiers {
    public static void main(String[] args) throws Exception {

        // 加载数据集
        int seed = 0;
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances train = trainTest[0];
        Instances test = trainTest[1];

        // 纯粹的weka:
        RandomForest randf = new RandomForest();
//        randf.setNumTrees(500);
        randf.setSeed(seed);

        // 在训练数据上训练RandomForest分类器
        randf.buildClassifier(train);                                   //也就是fit, train

        double acc = .0;
        for (Instance testInst : test) {
            // 使用RandomForest分类器对测试实例进行分类
            double pred = randf.classifyInstance(testInst);             //也就是predict
            // 获取测试实例的类分布
            double [] dist = randf.distributionForInstance(testInst); //也就是predict_proba

            // 如果预测的类与实际的类相同，就增加准确性
            if (pred == testInst.classValue())
                acc++;
        }

        // 计算最终的准确性
        acc /= test.numInstances();
        // 打印RandomForest分类器在ItalyPowerDemand数据集上的准确性
        System.out.println("Random Forest accuracy on ItalyPowerDemand: " + acc);

    }

}
