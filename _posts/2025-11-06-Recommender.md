---
layout: single
title: "Multi-faceted Movie Recommendation System"
excerpt: " Building Multi faceted Movies Recommender"
date: 2025-01-03
read_time: true
comments: true
share: true
related: true
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: "/assets/images/ml-project-banner.jpg"
  caption: "Exploring prediction with Random Forest"
class: wide
---

# Problem Statement

Have you ever felt overwhelmed by the millions of movies available online? I did, and I wanted to do something about it. I built a movie recommendation system that uses machine learning to suggest personalized movies based on your interests. My goal was to create a system that could help users discover new movies and enjoy their favorite films even more.


# Project Objectives
I aimed to create a movie recommendation engine that could deliver personalized suggestions through multiple approaches. Here's how I did it:

- *Demographic Filtering:* I built a system to showcase popular and highly-rated movies, categorized by genre, to give users a sense of what's trending.
- *Content-Based Filtering:* I designed a system to recommend movies similar to a given title, based on factors like plot, cast, crew, and keywords.
- *Collaborative Filtering:* I developed a model to predict user preferences by analyzing patterns in collective user ratings.
- *Hybrid Model:* I combined the strengths of content-based and collaborative filtering to create a powerful recommendation engine that considers both movie similarity and individual user taste.

By combining these approaches, I built a robust engine that can deliver personalized movie recommendations, helping users discover new favorites and enjoy their favorite films even more.


# Methodology & Technical Approach
I followed a structured approach to get the project up and running. Here's what I did:

- *Data Acquisition:* I sourced data from Kaggle's The Movies Dataset, using files like `movies_metadata.csv`, `credits.csv`, `keywords.csv`, `ratings_small.csv`, and `links_small.csv`.
- *Data Preprocessing:* I cleaned and prepared the data by:
    - Handling missing values
    - Converting data types for consistency
    - Merging datasets into a unified dataframe for analysis

This gave me a solid foundation to build my movie recommendation engine.


**a. Demographic Filtering (Genre & Popularity):**

To create a baseline for popular movies, I built a demographic filtering system that uses IMDB's weighted rating formula. This formula balances a movie's average rating (R) with the total number of votes (v) to ensure statistical significance.

I then created a function that uses this formula to generate ranked lists of top movies for any selected genre, giving users a sense of what's popular and highly-rated within their preferred genre.

**b. Content-Based Filtering:**
To recommend movies based on their intrinsic qualities, I developed two content-based models. First, I engineered a plot-based system using TF-IDF Vectorization on movie overviews and taglines to measure textual similarity. To create a more nuanced system, I built an enhanced metadata model by creating a feature "soup" from the top-billed cast, director, and keywords. After processing these keywords through stemming and relevance filtering, I applied Count Vectorization to build a powerful similarity matrix. I then encapsulated this logic into a function that returns a list of the most similar movies for any given title.

**b. Collaborative Filtering (Model-Based):**

To incorporate user behavior into the recommendations, I built a collaborative filtering system. I implemented the Singular Value Decomposition (SVD) algorithm using the Surprise library. After training the model on the ratings_small dataset, I rigorously validated its performance through 5-fold cross-validation, achieving a strong RMSE of ~0.897, confirming its predictive power. The final model is capable of accurately predicting how a specific user would rate any movie in the database.

**e. Hybrid Recommendation Engine:**

I developed a core function that synergizes the content-based and collaborative methods.
For a given user and movie title, the engine I built:

Uses my content-based model to find a list of similar movies.
Uses my trained SVD model to predict that specific user's rating for each of these similar movies.
Ranks the final list by these predicted ratings and returns the top 10 personalized recommendations.

# Key Results & Outputs

- Demographic System: 
I validated this system by having it generate accurate lists of top-rated films like Inception and The Dark Knight, as well as curated genre charts, such as ranking Forrest Gump among the top Romance films.
- Content-Based System:
The effectiveness of my model was clear when querying The Dark Knight returned a highly relevant list including direct sequels like The Dark Knight Rises and thematically similar films from the same director, like The Prestige.
- Hybrid System:
The power of my integrated approach was proven when, for a specific user and the movie Avatar, the engine delivered a personalized list of sci-fi/action classics like Aliens and Terminator 2, which the SVD model predicted that particular user would rate highly.

# Technologies Used
- Programming Language: Python
- Libraries & Frameworks: I applied Pandas, NumPy, Scikit-learn, Surprise (SVD), and NLTK to handle everything from data processing to model deployment.
- Core Techniques: I successfully implemented techniques like TF-IDF Vectorization, Cosine Similarity, and Matrix Factorization (SVD) to power the different recommendation models.


# Conclusion

In this project, I demonstrated my comprehensive understanding of modern recommendation system paradigms. By implementing and integrating Demographic, Content-Based, Collaborative, and Hybrid techniques, I built a powerful and versatile movie recommendation engine. This system effectively addresses the initial problem of content discovery, as it is capable of serving both general trending content and deeply personalized suggestions, thereby significantly enhancing the user experience on any movie-based platform.

---
