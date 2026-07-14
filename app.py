import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide"
)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("single_genre_artists.csv")

df = load_data()

# -----------------------------------------------------
# Feature Columns
# -----------------------------------------------------

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

st.sidebar.title("🎵 Amazon Music")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 EDA",
        "⚖ Feature Scaling",
        "🤖 K-Means",
        "📈 PCA",
        "📋 Cluster Summary",
        "ℹ About"
    ]
)

# -----------------------------------------------------
# HOME
# -----------------------------------------------------

if page == "🏠 Home":

    st.title("🎵 Amazon Music Clustering")

    st.markdown("""
### Project Overview

This project groups similar songs based on their audio features using Machine Learning.

### Objectives

- Discover similar songs
- Build song clusters
- Improve playlist generation
- Support music recommendation systems
""")

    st.info("Navigate using the sidebar to explore the project.")

# -----------------------------------------------------
# DATASET
# -----------------------------------------------------

elif page == "📂 Dataset":

    st.title("📂 Dataset")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.write(df.duplicated().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

# -----------------------------------------------------
# EDA
# -----------------------------------------------------

elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    feature = st.selectbox(
        "Select Feature",
        numeric_columns
    )

    st.subheader("Histogram")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(
        df[feature],
        kde=True,
        color="skyblue",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Box Plot")

    fig, ax = plt.subplots(figsize=(8,2))

    sns.boxplot(
        x=df[feature],
        color="orange",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df[numeric_columns].corr(),
        cmap="coolwarm",
        annot=False
    )

    st.pyplot(fig)

# -----------------------------------------------------
# FEATURE SCALING
# -----------------------------------------------------

elif page == "⚖ Feature Scaling":

    st.title("⚖ Feature Scaling")

    X = df[features]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=features
    )

    st.success("Feature Scaling Completed Successfully")

    st.dataframe(scaled_df.head())
    # -----------------------------------------------------
# K-MEANS CLUSTERING
# -----------------------------------------------------

elif page == "🤖 K-Means":

    st.title("🤖 K-Means Clustering")

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=4
    )

    model = KMeans(n_clusters=k, random_state=42)

    df["Cluster"] = model.fit_predict(X_scaled)

    st.success("Clustering Completed Successfully!")

    st.subheader("Clustered Dataset")

    if "name_song" in df.columns:
        st.dataframe(df[["name_song", "Cluster"]].head(20))
    else:
        st.dataframe(df.head(20))

    st.subheader("Cluster Count")

    st.bar_chart(df["Cluster"].value_counts().sort_index())


# -----------------------------------------------------
# PCA VISUALIZATION
# -----------------------------------------------------

elif page == "📈 PCA":

    st.title("📈 PCA Visualization")

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=4, random_state=42)

    clusters = model.fit_predict(X_scaled)

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cluster": clusters
    })

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        ax=ax
    )

    ax.set_title("PCA Cluster Visualization")

    st.pyplot(fig)


# -----------------------------------------------------
# CLUSTER SUMMARY
# -----------------------------------------------------

elif page == "📋 Cluster Summary":

    st.title("📋 Cluster Summary")

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=4, random_state=42)

    df["Cluster"] = model.fit_predict(X_scaled)

    summary = df.groupby("Cluster")[features].mean()

    st.subheader("Average Feature Values")

    st.dataframe(summary)

    st.subheader("Cluster Heatmap")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.heatmap(
        summary,
        annot=True,
        cmap="YlGnBu",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Clustered Dataset",
        data=csv,
        file_name="Amazon_Music_Clustered.csv",
        mime="text/csv"
    )


# -----------------------------------------------------
# ABOUT
# -----------------------------------------------------

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
### 🎵 Amazon Music Clustering

This project groups songs based on similar audio characteristics using the **K-Means Clustering** algorithm.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

### Machine Learning Concepts

- Data Cleaning
- Exploratory Data Analysis
- Feature Scaling
- K-Means Clustering
- PCA Visualization

### Business Applications

- Music Recommendation
- Playlist Generation
- Artist Analysis
- Music Discovery

---

**Developed by:** Logamithran BS
""")