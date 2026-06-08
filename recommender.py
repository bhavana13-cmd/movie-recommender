import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------
# Title
# ---------------------------
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on plot similarity!")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")

    movies = movies[['title', 'overview']]

    movies['overview'] = movies['overview'].fillna('')

    return movies

movies = load_data()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📊 Project Information")

st.sidebar.write(f"Total Movies: {len(movies)}")

st.sidebar.write(
    """
    **Algorithm Used**
    - TF-IDF Vectorization
    - Cosine Similarity
    """
)

# ---------------------------
# Create Similarity Matrix
# ---------------------------
@st.cache_resource
def create_similarity():

    tfidf = TfidfVectorizer(
        stop_words='english'
    )

    tfidf_matrix = tfidf.fit_transform(
        movies['overview']
    )

    similarity = cosine_similarity(
        tfidf_matrix
    )

    return similarity

similarity = create_similarity()

# ---------------------------
# User Input
# ---------------------------
st.subheader("🔍 Search Movie")

movie_name = st.text_input(
    "Type a Movie Name"
)

selected_movie = st.selectbox(
    "Or Select from List",
    movies['title'].sort_values()
)

# ---------------------------
# Recommendation Button
# ---------------------------
if st.button("🎥 Recommend"):

    if movie_name.strip():

        matches = movies[
            movies['title']
            .str.lower()
            .str.contains(
                movie_name.lower(),
                na=False
            )
        ]

        if matches.empty:

            st.error("❌ Movie not found!")

            st.stop()

        idx = matches.index[0]

    else:

        idx = movies[
            movies['title']
            == selected_movie
        ].index[0]

    selected_title = movies.iloc[idx]['title']

    similarity_scores = list(
        enumerate(similarity[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    st.success(
        f"Recommendations for: {selected_title}"
    )

    st.markdown("---")

    for i, (
        movie_index,
        score
    ) in enumerate(
        similarity_scores[1:6],
        start=1
    ):

        title = movies.iloc[
            movie_index
        ]['title']

        overview = movies.iloc[
            movie_index
        ]['overview']

        st.subheader(
            f"{i}. {title}"
        )

        st.write(overview)

        st.write(
            f"⭐ Similarity Score: {score:.2f}"
        )

        st.markdown("---")