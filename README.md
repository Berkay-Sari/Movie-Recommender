# Movie Recommender System

This repository contains a Movie Recommender System built using PySpark and the Alternating Least Squares (ALS) algorithm. It reads user ratings from CSV files and provides movie recommendations for both new and existing users.

## What It Does

- **New User Recommendations**: Allows new users to rate movies and receive personalized recommendations.
- **Existing User Recommendations**: Provides movie recommendations for existing users based on their past ratings.

## Data

The required CSV files can be downloaded from [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

To run the recommender system, simply prepare the required data files (`ratings.csv`, `links.csv`, `movies_metadata.csv`) and execute `main.py` to interact with the command-line menu.
