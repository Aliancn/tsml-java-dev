package examples_yy;

import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Examples to show different ways of loading and basic handling of datasets
 *
 * @author James Large (james.large@uea.ac.uk)
 *
 */
public class Ex01_yy_Datahandling {

    public static void main(String[] args) throws Exception {

        // We'll be loading the ItalyPowerDemand dataset which is distributed with this codebase
        int seed = 1;

        Instances train;
        Instances test;
        Instances[] trainTest;

        // Data load from the pre-defined split
        trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        train = trainTest[0];
        test = trainTest[1];


        //////////// Data inspection and handling:
        // We can look at the basic meta info

        System.out.println("train.relationName() = " + train.relationName());
        System.out.println("train.numInstances() = " + train.numInstances());
        System.out.println("train.numAttributes() = " + train.numAttributes());
        System.out.println("train.numClasses() = " + train.numClasses());

        // And the individual instances
        for (Instance inst : train)
            System.out.print(inst.classValue() + ", ");
        System.out.println("\n");

        // 输出train的前一个instance
        System.out.println(train.instance(0));
    }

}
