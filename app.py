
# Assignment 20: Building & Deploying a Recommendation System
# Content-Based Movie Recommendation System

import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "tmdb_5000_movies.csv"
TOP_N_DEFAULT = 5


# ==========================================================
# TASK 1: LOAD AND UNDERSTAND DATASET
# ==========================================================

@st.cache_data
def load_dataset(path):
    data = pd.read_csv(path)

    required_columns = ["title", "overview", "genres"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"Required column '{column}' is missing. "
                f"Available columns: {list(data.columns)}"
            )

    return data


# ==========================================================
# TASK 2: TEXT PREPROCESSING
# ==========================================================

def clean_text(text):
    """
    Convert text to lowercase, remove punctuation and special
    characters, remove English stopwords and normalize spaces.
    Missing values are replaced with empty strings.
    """
    if pd.isna(text):
        text = ""

    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(words)


def prepare_dataset(data):
    data = data.copy()

    # Handle missing values in text columns
    for column in ["overview", "genres"]:
        data[column] = data[column].fillna("")

    # Create a combined text field for recommendations
    data["recommendation_text"] = (
        data["overview"].astype(str)
        + " "
        + data["genres"].astype(str)
    )

    # Store cleaned text in the required clean_text column
    data["clean_text"] = data["recommendation_text"].apply(
        clean_text
    )

    # Remove records without usable descriptions
    data = data[data["clean_text"].str.strip() != ""].copy()

    # Remove duplicate movie titles
    data = data.drop_duplicates(
        subset=["title"],
        keep="first"
    ).reset_index(drop=True)

    return data


# ==========================================================
# TASK 3: TF-IDF VECTORIZATION
# ==========================================================

@st.cache_data
def create_tfidf_matrix(clean_texts):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(clean_texts)

    return tfidf_matrix, vectorizer.get_feature_names_out()


# ==========================================================
# TASK 4: COSINE SIMILARITY
# ==========================================================

@st.cache_data
def create_similarity_matrix(_tfidf_matrix):
    # Sparse output reduces memory usage for the similarity matrix.
    similarity_matrix = cosine_similarity(
        _tfidf_matrix,
        dense_output=False
    )

    return similarity_matrix


# ==========================================================
# TASK 5: BUILD RECOMMENDATION FUNCTION
# ==========================================================

def recommend(item_name, top_n=TOP_N_DEFAULT):
    """
    Return the top N movies similar to the selected movie.
    """
    if item_name not in title_to_index:
        return pd.DataFrame()

    item_index = title_to_index[item_name]

    # Find similarity scores for the selected item
    scores = similarity_matrix.getrow(item_index).toarray().ravel()

    # Sort scores from highest to lowest
    ranked_indices = scores.argsort()[::-1]

    recommendations = []
    for index in ranked_indices:
        # Exclude the selected movie itself
        if index == item_index:
            continue

        recommendations.append(index)

        if len(recommendations) >= top_n:
            break

    result = movies.iloc[recommendations][
        ["title", "overview", "genres"]
    ].copy()

    result["similarity_score"] = [
        float(scores[index]) for index in recommendations
    ]

    return result


# ==========================================================
# STREAMLIT APP: TASK 6
# ==========================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write(
    "Find movies similar to your selected movie using "
    "content-based filtering, TF-IDF and cosine similarity."
)

st.caption(
    "Dataset: TMDB 5000 Movie Dataset | Source: Kaggle"
)

try:
    raw_data = load_dataset(DATASET_PATH)
    movies = prepare_dataset(raw_data)

    # Task 1: display essential dataset information
    with st.expander("Task 1: Dataset Information"):
        st.write("Dataset shape:", raw_data.shape)
        st.write("Column names:", raw_data.columns.tolist())

        st.write("First 5 rows:")
        st.dataframe(raw_data.head(5), use_container_width=True)

        st.write("Missing values:")
        st.dataframe(
            raw_data.isnull().sum().rename("Missing Values"),
            use_container_width=True
        )

    # Task 2: display cleaned text
    with st.expander("Task 2: Text Preprocessing"):
        st.write("Cleaned dataset shape:", movies.shape)
        st.dataframe(
            movies[["title", "clean_text"]].head(5),
            use_container_width=True
        )

    # Task 3: vectorize the text
    tfidf_matrix, vocabulary = create_tfidf_matrix(
        tuple(movies["clean_text"].tolist())
    )

    st.write("### TF-IDF Vectorization")
    st.write("TF-IDF matrix shape:", tfidf_matrix.shape)
    st.write("Vocabulary size:", len(vocabulary))

    # Task 4: compute similarity between all movies
    similarity_matrix = create_similarity_matrix(tfidf_matrix)

    st.write("Similarity matrix shape:", similarity_matrix.shape)

    # Map movie titles to row indices
    title_to_index = {
        title: index
        for index, title in enumerate(movies["title"])
    }

    # Task 5 and Task 6: recommendation interface
    st.divider()
    st.subheader("Find Similar Movies")

    movie_titles = movies["title"].tolist()

    selected_movie = st.selectbox(
        "Select a movie",
        options=movie_titles
    )

    top_n = st.selectbox(
        "Number of recommendations",
        options=[3, 5, 10],
        index=1
    )

    if st.button("Recommend Movies", type="primary"):
        recommendations = recommend(selected_movie, top_n)

        if recommendations.empty:
            st.warning("No recommendations could be generated.")
        else:
            st.success(
                f"Movies similar to: {selected_movie}"
            )

            for _, row in recommendations.iterrows():
                with st.container(border=True):
                    st.subheader(row["title"])
                    st.write(
                        "Similarity score:",
                        f"{row['similarity_score']:.3f}"
                    )

                    if row["genres"]:
                        st.write("Genres:", row["genres"])

                    if row["overview"]:
                        st.write("Overview:", row["overview"])

    # Task 5: test recommendation function with 3 items
    with st.expander("Task 5: Test with 3 Different Movies"):
        test_titles = movie_titles[:3]

        for test_title in test_titles:
            st.write(f"**Input movie:** {test_title}")

            test_result = recommend(test_title, top_n=3)

            if test_result.empty:
                st.write("No recommendations available.")
            else:
                st.dataframe(
                    test_result[
                        ["title", "similarity_score"]
                    ],
                    use_container_width=True
                )

except FileNotFoundError:
    st.error(
        f"Dataset file '{DATASET_PATH}' was not found. "
        "Place tmdb_5000_movies.csv in the same folder as app.py."
    )

except (ValueError, KeyError) as error:
    st.error(f"Unable to process the dataset: {error}")
