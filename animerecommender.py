#!pip install pyspark
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql import Row
from pyspark.sql.types import (StructField, LongType, 
                               DoubleType, FloatType,
                               StructType, ArrayType,
                               IntegerType, StringType
                              )
import pyspark.sql.functions as f
from pyspark.sql.functions import lit



appName = "Anime_Recommendation_Engine_Spark"
master = "local"
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

class RecommendationEngine:
    """
    Anime Recommendation Engine
    """
    def __init__(
        self,
        appName="Anime_Recommendation",
        master="local",
        sc = sc,
        sqlContext = sqlContext,
        given_seed=607,
        files_directory="../input/"
        ):
            self.seed(given_seed)
            self.given_seed = given_seed
            self.appName = appName,
            self.master = master,
            self.directory = files_directory
            self.sc, self.sqlContext = sc, sqlContext
            self.model = None
            self.valid_data = None
            self.user_file_data = None
            self.anime_score_data = None
            self.anime_id_data = None
            self.training = None
            self.test = None
            self.predictions = None
            self.titles_dict = None
            self.valid_count = 0
            self.ufd_count = 0
            self.asd_count = 0
            self.aid_count = 0
            self.load_data()
            self.prep_and_clean_data()
            self.stratify_split()
            self.train_model()
            self.make_predictions()
            
    def seed(self,given_seed):
        print("Setting Seed...")
        random.seed(given_seed)
        np.random.seed(given_seed)
    def setSparkConfig(self):
        print("Setting Spark Config...")
        conf = SparkConf().setAppName(self.appName).setMaster(self.master)
        sc = SparkContext(conf=conf)
        sqlContext = SQLContext(sc)
        return sc, sqlContext
    def load_data(self):
        print("Loading Data...")
        user_file_path = self.directory + "users_cleaned.csv"
        anime_score_file_path = self.directory + "animelists_cleaned.csv"
        anime_id_file_path = self.directory + "anime_cleaned.csv"
        self.user_file_data = self.sqlContext.read.csv(
            user_file_path,
            header=True
            )
        self.anime_score_data = self.sqlContext.read.csv(
            anime_score_file_path,
            header=True
            )
        self.anime_id_data = self.sqlContext.read.csv(
            anime_id_file_path,
            header=True
        )
        print("Creating Dictionary for Titles")
        self.titles_dict = self.anime_id_data.rdd.map(lambda array: (int(array[0]),str(array[1]))).collectAsMap()
    def prep_and_clean_data(self):
        print("Preparing and Cleaning Data...")
        print("Getting Necessary Features")
        self.user_file_data = self.user_file_data.select(
            "user_id","username").withColumn("user_id", f.col("user_id").cast(IntegerType()))
        self.anime_score_data = self.anime_score_data.select("anime_id","username","my_score").withColumn(
            "anime_id", f.col("anime_id").cast(IntegerType())).withColumn(
                "my_score", f.col("my_score").cast(FloatType()))
        self.anime_id_data = self.anime_id_data.select("anime_id","title").withColumn(
            "anime_id", f.col("anime_id").cast(IntegerType()))
        print("Recording Initial Counts...")
        self.ufd_count, self.asd_count, self.aid_count = self.user_file_data.count(), self.anime_score_data.count(), self.anime_id_data.count()
        print(
            "User File Data: ",
            self.ufd_count,
            "Anime Score Data: ",
            self.asd_count,
            "Anime Id Data: ",
            self.aid_count
            )
        # Filtering out the nonsensical scores. According to Myanimelist website
        # the only valid scores are from 1-10
        print("Cleaning and Filtering Data...")
        self.anime_score_data = self.anime_score_data.filter(f.col("my_score")>0)
        self.asd_count = self.anime_score_data.count()
        initial_count = self.asd_count
        # Joining Anime Score Data and User File Data
        # to link user_id to every rating
        self.anime_score_data = self.anime_score_data.join(
            self.user_file_data,
            self.anime_score_data.username == self.user_file_data.username
            ).drop(self.anime_score_data.username).cache()
        # We want to then only consider anime series which have been rated
        # 5 or more times
        anime_score_data_counts = self.anime_score_data.groupBy("anime_id").count()
        multi_member_classes = anime_score_data_counts.filter(f.col("count")>=5)
        multi_mc_list = [row['anime_id'] for row in multi_member_classes.collect()]
        self.valid_data = self.anime_score_data.where(self.anime_score_data.anime_id.isin(multi_mc_list))
        self.valid_count = self.valid_data.count()
        print("Initial Data Count: ", initial_count, " Final Data Count: ", self.valid_count)
    def stratify_split(self):
        # We use stratifield sampling as our method to choose
        # our train and test sets. This way we have a pool of
        # users that exist on both sides. This allows us to form
        # predictions for all items that exist in the test set
        print("Splitting our Data into 80-20 Train and Test Sets...")
        fractions = self.valid_data.select("anime_id").distinct().withColumn(
            "fraction", lit(0.8)).rdd.collectAsMap()
        self.training = self.valid_data.stat.sampleBy("anime_id", fractions, self.given_seed)
        self.test = self.valid_data.subtract(self.training)
    def train_model(self):
        # Use alternating least scores matrix factorization
        # as the backbone of our recommendation engine
        print("Training our Model")
        als = ALS(
            rank=10,maxIter=25,regParam=0.01,
            userCol="user_id",itemCol="anime_id",
            ratingCol="my_score",seed=self.given_seed,implicitPrefs=True
            )
        self.model = als.fit(self.training)
    def make_predictions(self):
        # Making predicitons on our test data
        print("Making predictions")
        self.predictions = self.model.transform(self.test)
    def getUserPredictions(self,userId,records):
        print("Getting User ",userId,"'s predictions...")
        user_predictions = self.predictions.filter(f.col("user_id")==userId).sort("my_score",ascending=True)
        return user_predictions
    def getTopKRecs(self,userId,k):
        titles = self.titles_dict
        print("Getting top ",k," recommendations for User ",userId,"...")
        model_recommendations = self.model.recommendForAllUsers(k).filter(f.col("user_id")==userId)
        results = model_recommendations.first()
        print("Recommendations: ")
        recommendation_ids = []
        for result in results["recommendations"]:
            print(result["anime_id"],titles[result["anime_id"]],result["rating"])
            recommendation_ids.append(result["anime_id"])
        return recommendation_ids
    def getTopKRatedAnime(self,userId,k):
        # Gets the top rated items in the training set
        titles = self.titles_dict
        print("Getting User ",userId,"'s top ",k," rated anime...")
        user_rated_items = len(self.training.rdd.keyBy(lambda x: x["user_id"]).lookup(userId))
        print("User ",userId," rated ",user_rated_items," anime series.")
        key_ratings_by = self.training.rdd.keyBy(
            lambda x: x["user_id"]).sortBy(
                lambda x: x[1]["my_score"],ascending=False
                ).map(lambda x: (x[0],x[1]["anime_id"],titles[x[1]["anime_id"]],x[1]["my_score"]))
        topratings = key_ratings_by.filter(lambda x: x[0]==userId).map(lambda x: x[1]).take(10)
        return topratings
    def getItemVector(self,given_id):
        # Gets items(in this case) assocaited vector
        desired_vector= list(self.model.itemFactors.filter(
            f.col("id")==given_id).select(
                f.col("features")).first())
        return desired_vector
    def evaluate_recommendations(self,anime_ids,rec_ids):
        # Cosine Similarity is used so we can compare how similar
        # two items [Anime series in this case are]
        # We use correlation coefficient as well
        overall_similarity = 0
        overall_correlation = 0
        id_count = 0
        if type(anime_ids)!=list:
            anime_ids = [anime_ids]
        for anime_id in anime_ids:
            for rec_id in rec_ids:
                animeVector = self.getItemVector(anime_id)
                compareVector = self.getItemVector(rec_id)
                overall_similarity = overall_similarity + cosine_similarity(animeVector,compareVector)[0][0]
                overall_correlation = overall_correlation + np.corrcoef(animeVector[0],compareVector[0])[1][0]
                id_count = id_count + 1
        print("Total Cosine Similarity: ",overall_similarity)
        print("Total Correlation Score: ",overall_correlation)
        print("Total Records Considered: ",id_count)
        print("Average Cosine Similarity: ",overall_similarity/id_count)
        print("Average Correlation Score: ",overall_correlation/id_count)
    def setCosine(self,row,given_id):
        titles_dict = self.titles_dict
        itemVector = self.getItemVector(given_id)
        return titles_dict[row["id"]],float(cosine_similarity([row["features"]],itemVector))
    def getTopRelatedAnime(self,anime_id):
        # Get top related anime and simultaneously
        # evaluate their performance
        titles_dict = self.titles_dict
        itemVector = self.getItemVector(anime_id)
        sortedSims = self.model.itemFactors.rdd.map(
            lambda x: (
                x["id"],
                float(cosine_similarity([x["features"]],itemVector))
                )
            )
        sortedSims = sortedSims.sortBy(lambda x: x[1],ascending=False)
        topRelated = sortedSims.map(lambda x: x[0]).take(10)
        self.evaluate_recommendations(anime_id,topRelated)
        return topRelated
    def getTopRelatedUsers(self,user_id):
        # Get top related users and simultaneously
        # evaluate their performance
        titles_dict = self.titles_dict
        userVector = self.getUserVector(user_id)
        sortedSims = self.model.userFactors.rdd.map(
            lambda x: (
                x["id"],
                float(cosine_similarity([x["features"]],userVector))
                )
            )
        print("First Record")
        print(sortedSims.take(1))
        sortedSims = sortedSims.sortBy(lambda x: x[1],ascending=False)
        topRelated = sortedSims.map(lambda x: x[0]).take(10)
        self.evaluate_user_recommendations(user_id,topRelated)
        return topRelated
    def getUserVector(self,given_id):
        # Gets items(in this case) assocaited vector
        desired_vector= list(self.model.userFactors.filter(
            f.col("id")==given_id).select(
                f.col("features")).first())
        return desired_vector
    def evaluate_user_recommendations(self,user_ids,rec_ids):
        # Cosine Similarity is used so we can compare how similar
        # two items [Anime series in this case are]
        # We use correlation coefficient as well
        overall_similarity = 0
        overall_correlation = 0
        id_count = 0
        if type(user_ids)!=list:
            user_ids = [user_ids]
        for user_id in user_ids:
            for rec_id in rec_ids:
                userVector = self.getUserVector(user_id)
                compareVector = self.getUserVector(rec_id)
                overall_similarity = overall_similarity + cosine_similarity(userVector,compareVector)[0][0]
                overall_correlation = overall_correlation + np.corrcoef(userVector[0],compareVector[0])[1][0]
                id_count = id_count + 1
        print("Total Cosine Similarity: ",overall_similarity)
        print("Total Correlation Score: ",overall_correlation)
        print("Total Records Considered: ",id_count)
        print("Average Cosine Similarity: ",overall_similarity/id_count)
        print("Average Correlation Score: ",overall_correlation/id_count)
        
        
# Testing our recommendation engine
anime_recommender_engine = RecommendationEngine()
user_preds = anime_recommender_engine.getUserPredictions(56741,2)
user_preds.take(2)
top10recs = anime_recommender_engine.getTopKRecs(1645,10)
print(top10recs)
top10anime = anime_recommender_engine.getTopKRatedAnime(1645,10)
print(top10anime)
anime_recommender_engine.evaluate_recommendations(top10anime,top10recs)
top10related = anime_recommender_engine.getTopRelatedAnime(1)
print(top10related)
top10relatedusers = anime_recommender_engine.getTopRelatedUsers(2255153)
print(top10relatedusers)