import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import threading

sc = SparkContext('local[5]', 'Movie Recommender')
sc.setLogLevel("ERROR")
spark = SparkSession(sc)
class _RecommenderModel:
    def __init__(self) -> None:
        self.data_schema = StructType([
            StructField('userId', IntegerType(), False),
            StructField('movieId', IntegerType(), False),
            StructField('rating', FloatType(), False),
            StructField('timestamp',IntegerType(), False)
        ])
        self.df = spark.read.csv(
            'ratings_small.csv', header=True, schema=self.data_schema
        ).cache()
        self.thread = None
        self.model = self._train()
        # self._learn()
        
    def _learn(self):
        self.thread = threading.Thread(target=self._learn_contigious)
        self.thread.daemon = True
        self.thread.start()
    
    def _learn_contigious(self):
        while True:
            self.model = self._train()
        
    def _train(self):
        ratings = self.df.select(
            'userId',
            'movieId',
            'rating'
        )

        (training, test) = ratings.randomSplit([0.7, 0.3], seed=42)

        als = ALS(
                rank=30,
                maxIter=4, 
                regParam=0.1,
                userCol='userId', 
                itemCol='movieId', 
                ratingCol='rating',
                coldStartStrategy='drop',
                implicitPrefs=False
                )
        return als.fit(training)
    
    def recommend_for_users(self, users, numItems=1):
        return self.model.recommendForUserSubset(self.df.filter(sql_func.col("userId").isin(users)), numItems=numItems)
    
    def add_new_record(self, movieId, rating):
        userId = 3333089
        new_rating_row = Row(userId=userId, movieId=movieId, rating=rating, timestamp=0)
        self.df = self.df.union(spark.createDataFrame([new_rating_row]))
    def recommend_for_new_user(self, user, numItems=1):
        self.model = self._train()
        return self.model.recommendForUserSubset(self.df.filter(self.df.userId == user), numItems=numItems)
    
Recommender = _RecommenderModel()
