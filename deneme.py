from RecommenderModel import Recommender, spark
from pyspark.sql.functions import col, explode
import time

# TODO: user 270896 rates movies
# 1 toy story
# 59784 kung fu panda
# 260 star wars
# 270896,1,4.8,1074789724
# 270896,59784,5,1074799724
# 270896,260,4.6,1074989724

links_df = spark.read.csv("links.csv", header=True)
metadata_df = spark.read.csv("movies_metadata.csv", header=True)

links_df = links_df.withColumn("movieId", col("movieId").cast("int"))

joined_df = links_df.join(metadata_df, links_df.tmdbId == metadata_df.id, "inner")

reverse_mapping_df = joined_df.select("original_title", "movieId")

def new_user():
    print("Creating a new user...")

    num_movies = int(input("How many movies would you like to rate?"))

    for _ in range(num_movies):
        while True:
            movie_name = input("Enter the original name of the movie: ")
            movie_id = reverse_mapping_df.filter(col("original_title") == movie_name).select("movieId").first()
                    
            if movie_id:         
                rating = int(input("How would you rate this movie out of 10: ")) / 2
                Recommender.add_new_record(movie_id['movieId'], rating)
                print(f"Rated '{movie_name}': {rating} (#{movie_id['movieId']})")
                break  
            else:
                print(f"No movie found with the name '{movie_name}'. Please enter a valid movie name.")
    
    num_items = int(input("How many movie do you want to recommend? "))
    recommendeds = Recommender.recommend_for_new_user(3333089, num_items)
    exploded_df = recommendeds.select("userId", explode("recommendations").alias("rec"))

    exploded_df = exploded_df.select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("rating"))

    result_df = exploded_df.join(joined_df.select("movieId", "original_title"), exploded_df.movieId == joined_df.movieId,"inner") \
    .select(col("original_title").alias("title"), "rating")


    result_df.show(truncate=False)
def existing_users():
    print("Accessing existing users...")
    choosen_users = input("Enter a list of user ids separated by spaces: ")
    user_list = choosen_users.split()
    users = [int(num) for num in user_list]
    
    num_items = int(input("How many movie do you want to recommend for each user? "))

    recommendeds = Recommender.recommend_for_users(users, num_items)

    exploded_df = recommendeds.select("userId", explode("recommendations").alias("rec"))

    exploded_df = exploded_df.select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("rating"))

    result_df = exploded_df.join(joined_df.select("movieId", "original_title"), exploded_df.movieId == joined_df.movieId, "inner") \
        .select(exploded_df.userId, col("original_title").alias("title"), "rating")

    result_df.show(truncate=False)

def main():
    while True:
        print("Menu:")
        print("1. New User")
        print("2. Existing Users")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            new_user()
        elif choice == '2':
            existing_users()
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
