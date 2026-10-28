# Assignment 20: Building & Deploying a Recommendation System

## 1. Project Overview

This project implements a content-based movie recommendation system. It recommends movies with similar descriptions and genres using TF-IDF vectorization and cosine similarity.

The application provides a simple user interface built with Streamlit.

## 2. Dataset Information

- **Dataset Name:** TMDB 5000 Movie Dataset
- **Source:** Kaggle
- **Dataset Link:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- **CSV File:** `tmdb_5000_movies.csv`
- **Important Columns:** `title`, `overview`, `genres`

The movie overview and genre information are combined to create text features for recommendation.

## 3. Libraries Used

The project uses only the permitted external libraries:

- Pandas
- Scikit-learn
- Streamlit

Python's built-in `re` module is used for text cleaning.

## 4. Project Structure

```text
Assignment20/
│
├── app.py
├── requirements.txt
├── README.md
└── tmdb_5000_movies.csv
```

## 5. Tasks Completed

### Task 1: Load and Understand Dataset
Loads the Kaggle CSV using Pandas and displays the dataset shape, column names, first five rows and missing-value counts.

### Task 2: Text Preprocessing
Converts text to lowercase, removes punctuation and special characters, removes English stopwords, handles missing text and stores the result in `clean_text`.

### Task 3: TF-IDF Vectorization
Uses `TfidfVectorizer()` with `max_features=5000` and `ngram_range=(1, 2)` to convert movie text into numerical vectors.

### Task 4: Similarity Computation
Calculates cosine similarity between all movie vectors and stores the resulting similarity matrix.

### Task 5: Build Recommendation Function
Implements `recommend(item_name, top_n=5)`, which finds the selected movie, calculates similarity scores, sorts the results and returns similar movies. The function is tested with at least three different titles.

### Task 6: Build UI Using Streamlit
Provides a dropdown to select a movie, an option to select the number of recommendations, and a button to display recommended movies.

### Task 7: Git and GitHub Setup
Initializes a Git repository, creates a GitHub repository and pushes the project files to GitHub.

### Task 8: Deploy Application on Render
Connects the GitHub repository to Render and deploys the Streamlit application.

### Task 9: Final Validation
Checks the deployed URL, confirms that the application loads, tests movie selection and verifies that recommendations are displayed.

## 6. Install Dependencies

Run the following command in the VS Code terminal:

```bash
pip install -r requirements.txt
```

## 7. Run the Application Locally

Ensure `tmdb_5000_movies.csv` is in the same folder as `app.py`.

Run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 8. GitHub Setup

1. Create a new repository on GitHub.
2. Open the VS Code terminal in the project folder.
3. Initialize Git and push the project files:

```bash
git init
git add app.py requirements.txt README.md tmdb_5000_movies.csv
git commit -m "Add movie recommendation system"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

## 9. Render Deployment

1. Sign in to Render.
2. Create a new Web Service and connect the GitHub repository.
3. Configure the service with these commands:

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

4. Select a suitable available service plan and deploy the application.
5. Wait for the deployment to finish.
6. Open the generated public URL and test the recommendation system.

## 10. Conclusion

This project demonstrates a content-based recommendation system using text preprocessing, TF-IDF vectorization and cosine similarity. It provides an interactive Streamlit interface and a deployment workflow using GitHub and Render.

The system recommends movies based on their textual descriptions and genres. It does not use collaborative filtering or user ratings.
