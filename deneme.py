from RecommenderModel import Recommender, spark
from pyspark.sql.functions import col, explode, round
from difflib import get_close_matches

links_df = spark.read.csv("links.csv", header=True)
metadata_df = spark.read.csv("movies_metadata.csv", header=True)

links_df = links_df.withColumn("movieId", col("movieId").cast("int"))

joined_df = links_df.join(metadata_df, links_df.tmdbId == metadata_df.id, "inner")

reverse_mapping_df = joined_df.select("original_title", "movieId")

def get_closest_matches(movie_name, dataframe, threshold=0.7, num_suggestions=3):
    all_movie_names = dataframe.select("original_title").distinct().rdd.flatMap(lambda x: x).collect()

    similar_matches = get_close_matches(movie_name, all_movie_names, n=num_suggestions, cutoff=threshold)

    if len(similar_matches) < num_suggestions:
        exact_matches = [name for name in all_movie_names if movie_name.lower() in name.lower()]
        exact_matches = [match for match in exact_matches if match not in similar_matches]
        matches = list(set(similar_matches + exact_matches))
    else:
        matches = similar_matches

    return matches[:num_suggestions]

def new_user():
    print("Creating a new user...")

    num_movies = int(input("How many movies would you like to rate: "))

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
                print(f"No movie found with the name '{movie_name}'.")
                suggestions = get_closest_matches(movie_name, reverse_mapping_df)
                
                if(len(suggestions) > 0): 
                    print(f"Did you mean one of these? {', '.join(suggestions)}")

    num_items = int(input("How many movie do you want to recommend: "))
    recommendeds = Recommender.recommend_for_new_user(num_items)
    exploded_df = recommendeds.select("userId", explode("recommendations").alias("rec"))

    exploded_df = exploded_df.select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("rating"))

    result_df = exploded_df.join(joined_df.select("movieId", "original_title"), exploded_df.movieId == joined_df.movieId,"inner") \
    .select(col("original_title").alias("title"), "rating")
    
    result_df = result_df.withColumn("rating", round(col("rating"), 1))
    
    result_df = result_df.orderBy(result_df.rating.desc())
    result_df.show(truncate=False)
    
def existing_users():
    print("Accessing existing users...")
    choosen_users = input("Enter a list of user ids separated by spaces: ")
    user_list = choosen_users.split()
    users = [int(num) for num in user_list]
    
    num_items = int(input("How many movie do you want to recommend for each user: "))

    recommendeds = Recommender.recommend_for_users(users, num_items)
    exploded_df = recommendeds.select("userId", explode("recommendations").alias("rec"))

    exploded_df = exploded_df.select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("rating"))

    result_df = exploded_df.join(joined_df.select("movieId", "original_title"), exploded_df.movieId == joined_df.movieId, "inner") \
        .select(exploded_df.userId, col("original_title").alias("title"), "rating")
    
    result_df = result_df.withColumn("rating", round(col("rating"), 1))
    
    user_ids = result_df.select("userId").distinct().collect()

    for user_id in user_ids:
        user_result_df = result_df.filter(col("userId") == user_id.userId)
        print(f"Results for User ID {user_id.userId}:")
        user_result_df = user_result_df.orderBy(user_result_df.rating.desc())
        user_result_df.show(truncate=False)

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
