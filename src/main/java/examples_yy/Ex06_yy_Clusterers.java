package examples_yy;

import experiments.data.DatasetLoading;
import tsml.clusterers.UnsupervisedShapelets;
import utilities.ClusteringUtilities;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.util.Arrays;

import static utilities.InstanceTools.deleteClassAttribute;

/**
 * 展示构建聚类器和基本使用方法的示例。
 */
public class Ex06_yy_Clusterers {
    public static void main(String[] args) throws Exception {

        // 随机数生成的种子
        int seed = 0;
        // 加载ItalyPowerDemand数据集并将其分为训练集和测试集
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances inst = trainTest[0];
        Instances inst2 = trainTest[1];
        // 合并训练集和测试集
        inst.addAll(inst2);

        // 创建一个UnsupervisedShapelets聚类器对象，并设置聚类的数量
        UnsupervisedShapelets us = new UnsupervisedShapelets();
        us.setNumberOfClusters(inst.numClasses());
        // 使用合并的数据集构建聚类器
        us.buildClusterer(inst);

        // 通过调用getAssignments()方法，可以找到每个数据实例的聚类分配。
        // assignments数组的索引将匹配Instances对象，即索引0值为1 == 数据的第一个实例
        // 分配给聚类1。
        int[] tsAssignments = us.getAssignments();
        System.out.println("UnsupervisedShapelets聚类分配：");
        System.out.println(Arrays.toString(tsAssignments));

        // 输出分类结果
        for (int i = 0; i < 10; i++) {
            System.out.println("第" + i + "个数据实例的聚类分配为：" + tsAssignments[i]);
        }

        // Rand指数是评估聚类的流行度量。有一个实用方法可以计算
        // 这个。
        double tsRandIndex = ClusteringUtilities.randIndex(tsAssignments, inst);
        System.out.println("UnsupervisedShapelets Rand指数：");
        System.out.println(tsRandIndex);

        // weka也实现了一系列的聚类算法。在使用之前，必须删除任何类值。
        Instances copy = new Instances(inst);
        deleteClassAttribute(copy);
        SimpleKMeans km = new SimpleKMeans();
        km.setNumClusters(inst.numClasses());
        km.setPreserveInstancesOrder(true);
        // 使用复制的数据集构建聚类器
        km.buildClusterer(copy);

        // 从SimpleKMeans聚类器获取聚类分配
        int[] wekaAssignments = km.getAssignments();
        System.out.println("SimpleKMeans聚类分配：");
        System.out.println(Arrays.toString(wekaAssignments));

        // 计算SimpleKMeans聚类器的Rand指数
        double wekaRandIndex = ClusteringUtilities.randIndex(wekaAssignments, inst);
        System.out.println("SimpleKMeans Rand指数：");
        System.out.println(wekaRandIndex);
    }
}
