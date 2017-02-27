import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jfree.data.category.DefaultCategoryDataset
import org.joda.time.{DateTime, Duration}
/**
  * Created by 08121 on 2017/2/27.
  */
object RunNaiveBayesBinary {
  def main(args: Array[String]): Unit = {

    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("RunNaiveBayesBinary")
    println("==========数据准备阶段===============")

    val (trainData, validationData, testData, categoriesMap) = PrepareData(sc)

    trainData.persist(); validationData.persist(); testData.persist()
    println("==========训练评估阶段===============")

    println()
    print("是否需要进行参数调校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData)
      println("==========测试阶段===============")
      val auc = evaluateModel(model, testData)
      println("使用testata测试最佳模型,结果 AUC:" + auc)
      println("==========预测数据===============")
      PredictData(sc, model, categoriesMap)
    } else {
      val model = trainEvaluate(trainData, validationData)
      println("==========测试阶段===============")
      val auc = evaluateModel(model, testData)
      println("使用testata测试最佳模型,结果 AUC:" + auc)
      println("==========预测数据===============")
      PredictData(sc, model, categoriesMap)

    }

    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], Map[String, Int]) = {
    //----------------------1.导入转换数据-------------
    print("开始导入数据...")
    val rawDataWithHeader = sc.textFile("F:/Data Folder/train.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val lines = rawData.map(_.split("\t"))
    println("共计：" + lines.count.toString() + "条")
    //----------------------2.创建训练评估所需数据 RDD[LabeledPoint]-------------
    val categoriesMap = lines.map(fields => fields(3)).distinct.collect.zipWithIndex.toMap
    val labelpointRDD = lines.map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)
      val label = trFields(fields.size - 1).toInt
      LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
    }

    val featuresData = labelpointRDD.map(labelpoint => labelpoint.features)
    val stdScaler = new StandardScaler(withMean = false, withStd = true).fit(featuresData)
    val scaledRDD = labelpointRDD.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)))
    //----------------------3.以随机方式将数据分为3个部分并且返回-------------
    val Array(trainData, validationData, testData) = scaledRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("将数据分trainData:" + trainData.count() + "   validationData:" + validationData.count() + "   testData:" + testData.count())
    return (trainData, validationData, testData, categoriesMap) //返回数据
  }

  def PredictData(sc: SparkContext, model: NaiveBayesModel, categoriesMap: Map[String, Int]): Unit = {

    //----------------------1.导入转换数据-------------
    print("开始导入数据...")
    val rawDataWithHeader = sc.textFile("F:/Data Folder/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val lines = rawData.map(_.split("\t"))
    println("共计：" + lines.count.toString() + "条")
    //----------------------2.创建训练评估所需的数据 RDD[LabeledPoint]-------------
    val labelpointRDD = lines.map { fields =>
      val trimmed = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trimmed.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)
      val label = 0
      val url = trimmed(0)
      (LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures)), url)
    }
    val featuresRDD = labelpointRDD.map { case (labelpoint, url) => labelpoint.features }
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresRDD)
    val scaledRDD = labelpointRDD.map { case (labelpoint, url) => (LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)), url) }
    scaledRDD.take(10).map {
      case (labelpoint, url) =>
        val predict = model.predict(labelpoint.features)
        var predictDesc = { predict match { case 0 => "暂时性网页(ephemeral)"; case 1 => "长青网页(evergreen)"; } }
        println("网址：  " + url + "==>预测:" + predictDesc)
    }

  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): NaiveBayesModel = {
    print("开始训练...")
    val (model, time) = trainModel(trainData, 5)
    println("训练完成,所需时间:" + time + "毫秒")
    val AUC = evaluateModel(model, validationData)
    println("评估结果AUC=" + AUC)
    return (model)
  }

  def trainModel(trainData: RDD[LabeledPoint], lambda: Int): (NaiveBayesModel, Double) = {
    val startTime = new DateTime()
    val model = NaiveBayes.train(trainData, lambda)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }
  def evaluateModel(model: NaiveBayesModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val AUC = Metrics.areaUnderROC
    (AUC)
  }
  def predict(model: NaiveBayesModel, testData: RDD[LabeledPoint]): Unit = {
    println("最佳模型使用testData前10条数据进行预测:")
    val testPredictData = testData.take(10)
    testPredictData.foreach { test =>
      val predict = model.predict(test.features)
      val result = (if (test.label == predict) "正确" else "错误")
      println("实际结果:" + test.label + "预测结果:" + predict + result + test.features)
    }
  }

  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): NaiveBayesModel = {
    println("-----评估 lambda参数使用 1, 3, 5, 15, 25---------")
    evaluateParameter(trainData, validationData, "lambda", Array(1, 3, 5, 15, 25))

    println("-----所有参数交叉评估找出最好的参数组合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array(1, 3, 5, 15, 25))
    return (bestModel)
  }
  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint],
                        evaluateParameter: String, lambdaArray: Array[Int]) =
  {
    var dataBarChart = new DefaultCategoryDataset()
    var dataLineChart = new DefaultCategoryDataset()
    for (lambda <- lambdaArray) {
      val (model, time) = trainModel(trainData, lambda)
      val auc = evaluateModel(model, validationData)
      val parameterData =
        evaluateParameter match {
          case "lambda" => lambda;
        }
      dataBarChart.addValue(auc, evaluateParameter, parameterData.toString())
      dataLineChart.addValue(time, "Time", parameterData.toString())
    }
    //Chart.plotBarLineChart("NaiveBayes evaluations " + evaluateParameter, evaluateParameter, "AUC", 0.48, 0.7, "Time", dataBarChart, dataLineChart)
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], lambdaArray: Array[Int]): NaiveBayesModel =
  {
    val evaluations =
      for (lambda <- lambdaArray) yield {
        val (model, time) = trainModel(trainData, lambda)
        val auc = evaluateModel(model, validationData)
        (lambda, auc)
      }
    val BestEval = (evaluations.sortBy(_._2).reverse)(0)
    println("调校后最佳参数：lambda:" + BestEval._1 + "  ,结果AUC = " + BestEval._2)

    val bestModel = NaiveBayes.train(
      trainData.union(validationData), BestEval._1)
    return bestModel
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}
