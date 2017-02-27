import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.{DateTime, Duration}

/**
  * Created by 08121 on 2017/2/27.
  */
object RunDecisionTreeMulti {
  def main(args: Array[String]): Unit = {
    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("RunDecisionTreeMulti")
    println("==========数据准备阶段===============")
    val (trainData, validationData, testData) = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========训练评估阶段===============")

    println()
    print("是否需要进行参数调校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData)
      println("==========测试阶段===============")
      val precise = evaluateModel(model, testData)
      println("使用testata测试,结果 precise:" + precise)
      println("==========预测数据===============")
      PredictData(sc, model)
    } else {
      val model = trainEvaluate(trainData, validationData)
      println("==========测试阶段===============")
      val precise = evaluateModel(model, testData)
      println("使用testata测试,结果 precise:" + precise)
      println("==========预测数据===============")
      PredictData(sc, model)
    }
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //----------------------1.导入转换数据-------------
    print("开始导入数据...")
    val rawData = sc.textFile("F:/Data Folder/covtype.data")
    println("共计：" + rawData.count.toString() + "条")
    //----------------------2.创建训练评估所需数据 RDD[LabeledPoint]-------------
    println("准备训练数据...")
    val labelpointRDD = rawData.map { record =>
      val fields = record.split(',').map(_.toDouble)
      val label = fields.last - 1
      LabeledPoint(label, Vectors.dense(fields.init))
    }
    println( labelpointRDD.first())
    //----------------------3.以随机方式将数据分为3个部分并且返回-------------
    val Array(trainData, validationData, testData) = labelpointRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("将数据分为 trainData:" + trainData.count() + "   cvData:" + validationData.count() + "   testData:" + testData.count())
    return (trainData, validationData, testData)
  }

  def PredictData(sc: SparkContext, model: DecisionTreeModel): Unit = {
    val rawData = sc.textFile("F:/Data Folder/covtype.data")
    println("共计：" + rawData.count.toString() + "条")
    println("准备测试数据...")
    val Array(pData, oData) = rawData.randomSplit(Array(0.1, 0.9))
    val data = pData.take(20).map { record =>
      val fields = record.split(',').map(_.toDouble)
      val features = Vectors.dense(fields.init)
      val label = fields.last - 1
      val predict = model.predict(features)
      val result = (if (label == predict) "正确" else "错误")
      println("土地条件：海拔:" + features(0) + " 方位:" + features(1) + " 斜率:" + features(2) + " 水源垂直距离:" + features(3) + " 水源水平距离:" + features(4) + " 9点时阴影:" + features(5) + "....==>预测:" + predict + " 实际:" + label + "结果:" + result)
    }
  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    print("开始训练...")
    val (model, time) = trainModel(trainData, "entropy", 20, 100)
    println("训练完成,所需时间:" + time + "毫秒")
    val precision = evaluateModel(model, validationData)
    println("评估结果precision=" + precision)
    return (model)
  }

  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }

  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }

    val Metrics = new MulticlassMetrics(scoreAndLabels)
    val precision = Metrics.precision
    (precision)
  }
  def testModel(model: DecisionTreeModel, testData: RDD[LabeledPoint]): Unit = {
    val precise = evaluateModel(model, testData)
    println("使用testData测试,结果 precise:" + precise)
    println("最佳模型使用testData前50条数据进行预测:")
    val PredictData = testData.take(50)
    PredictData.foreach { data =>
      val predict = model.predict(data.features)
      val result = (if (data.label == predict) "正确" else "错误")
      println("实际结果:" + data.label + "预测结果:" + predict + result + data.features)
    }

  }

  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    println("-----评估 Impurity参数使用 gini, entropy---------")
    evaluateParameter(trainData, validationData, "impurity", Array("gini", "entropy"), Array(10), Array(10))
    println("-----评估MaxDepth参数使用 (3, 5, 10, 15, 20)---------")
    evaluateParameter(trainData, validationData, "maxDepth", Array("gini"), Array(3, 5, 10, 15, 20, 25), Array(10))
    println("-----评估maxBins参数使用 (3, 5, 10, 50, 100)---------")
    evaluateParameter(trainData, validationData, "maxBins", Array("gini"), Array(10), Array(3, 5, 10, 50, 100, 200))
    println("-----所有参数交叉评估找出最好的参数组合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array("gini", "entropy"),
      Array(3, 5, 10, 15, 20), Array(3, 5, 10, 50, 100))
    return (bestModel)
  }
  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint],
                        evaluateParameter: String, impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]) =
  {
    var dataBarChart = new DefaultCategoryDataset()
    var dataLineChart = new DefaultCategoryDataset()
    for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) {
      val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
      val precise = evaluateModel(model, validationData)
      val parameterData =
        evaluateParameter match {
          case "impurity" => impurity;
          case "maxDepth" => maxDepth;
          case "maxBins"  => maxBins
        }
      dataBarChart.addValue(precise, evaluateParameter, parameterData.toString())
      dataLineChart.addValue(time, "Time", parameterData.toString())
    }
    //Chart.plotBarLineChart("DecisionTree evaluations " + evaluateParameter, evaluateParameter, "precision", 0.6, 1, "Time", dataBarChart, dataLineChart)
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel =
  {
    val evaluationsArray =
      for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) yield {
        val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
        val precise = evaluateModel(model, validationData)
        (impurity, maxDepth, maxBins, precise)
      }
    val BestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
    println("调校后最佳参数：impurity:" + BestEval._1 + "  ,maxDepth:" + BestEval._2 + "  ,maxBins:" + BestEval._3
      + "  ,结果precise = " + BestEval._4)
    val (bestModel, time) = trainModel(trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3)
    return bestModel
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}
