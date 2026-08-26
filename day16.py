import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("\n========== DATA LOADED ==========")
print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

print("\nMovies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())


# ============================================================
# 2. RENAME COLUMNS
# ============================================================

movies = movies.rename(columns={
    "movieId": "movie_id"
})

ratings = ratings.rename(columns={
    "userId": "user_id",
    "movieId": "movie_id"
})


# ============================================================
# 3. CREATE MOVIE ID <-> TITLE MAPPING
# ============================================================

title_to_id = dict(
    zip(movies["title"], movies["movie_id"])
)

id_to_title = dict(
    zip(movies["movie_id"], movies["title"])
)


# ============================================================
# 4. CONTENT-BASED KNN
# ============================================================

print("\n========== CONTENT-BASED KNN ==========")

mlb = MultiLabelBinarizer()

genre_matrix = mlb.fit_transform(
    movies["genres"].fillna("").str.split("|")
)

print("Genre matrix shape:", genre_matrix.shape)


def content_based_recommend(title, k=5):

    if title not in title_to_id:
        print(f"Movie '{title}' not found.")
        return pd.DataFrame()

    movie_id = title_to_id[title]

    movie_position = movies.index[
        movies["movie_id"] == movie_id
    ][0]

    number_of_neighbors = min(
        k + 1,
        len(movies)
    )

    model = NearestNeighbors(
        n_neighbors=number_of_neighbors,
        metric="cosine"
    )

    model.fit(genre_matrix)

    distances, indices = model.kneighbors(
        [genre_matrix[movie_position]]
    )

    recommendations = []

    for index, distance in zip(
        indices[0],
        distances[0]
    ):

        if index == movie_position:
            continue

        recommendations.append({
            "title": movies.iloc[index]["title"],
            "genres": movies.iloc[index]["genres"],
            "similarity": round(
                1 - distance,
                3
            )
        })

        if len(recommendations) == k:
            break

    return pd.DataFrame(recommendations)


print("\nRecommendations similar to Toy Story:")

print(
    content_based_recommend(
        "Toy Story (1995)",
        k=5
    )
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Training ratings:", len(train))
print("Testing ratings:", len(test))


# ============================================================
# 6. CREATE TRAINING RATING MATRIX
# ============================================================

train_matrix = train.pivot_table(
    index="movie_id",
    columns="user_id",
    values="rating"
)

train_means = train_matrix.mean(axis=1)

train_centered = train_matrix.sub(
    train_means,
    axis=0
).fillna(0)

print("\nTraining matrix shape:")
print(train_centered.shape)


# ============================================================
# 7. USER RATING LOOKUP
# ============================================================

user_ratings_lookup = {}

for user_id, group in train.groupby("user_id"):

    user_ratings_lookup[user_id] = dict(
        zip(
            group["movie_id"],
            group["rating"]
        )
    )


# ============================================================
# 8. COLLABORATIVE FILTERING KNN
# ============================================================

def predict_rating(
    user_id,
    movie_id,
    k,
    matrix=train_centered,
    means=train_means,
    user_ratings=user_ratings_lookup
):

    # If movie does not exist in training data
    if movie_id not in matrix.index:
        return means.mean()

    number_of_neighbors = min(
        k + 1,
        len(matrix)
    )

    model = NearestNeighbors(
        n_neighbors=number_of_neighbors,
        metric="cosine"
    )

    model.fit(matrix.values)

    movie_position = matrix.index.get_loc(
        movie_id
    )

    distances, indices = model.kneighbors(
        [matrix.values[movie_position]]
    )

    neighbor_ids = []
    neighbor_distances = []

    for index, distance in zip(
        indices[0],
        distances[0]
    ):

        neighbor_id = matrix.index[index]

        if neighbor_id == movie_id:
            continue

        neighbor_ids.append(neighbor_id)
        neighbor_distances.append(distance)

        if len(neighbor_ids) == k:
            break

    # Ratings made by this user
    user_history = user_ratings.get(
        user_id,
        {}
    )

    weighted_sum = 0.0
    weight_total = 0.0

    for neighbor_id, distance in zip(
        neighbor_ids,
        neighbor_distances
    ):

        if neighbor_id in user_history:

            similarity = 1 - distance

            weighted_sum += (
                similarity *
                user_history[neighbor_id]
            )

            weight_total += abs(similarity)

    # If user has not rated any neighbors
    if weight_total == 0:

        prediction = means.get(
            movie_id,
            means.mean()
        )

    else:

        prediction = (
            weighted_sum /
            weight_total
        )

    # Keep prediction between 0.5 and 5
    prediction = np.clip(
        prediction,
        0.5,
        5.0
    )

    return prediction


# ============================================================
# 9. TEST DIFFERENT K VALUES
# ============================================================

print("\n========== K VALUE COMPARISON ==========")

K_VALUES = [
    1,
    3,
    5,
    10,
    15,
    20
]

results = []


for k in K_VALUES:

    print(f"\nTesting K = {k}")

    predictions = []
    actual_values = []

    for _, row in test.iterrows():

        prediction = predict_rating(
            user_id=row["user_id"],
            movie_id=row["movie_id"],
            k=k
        )

        predictions.append(prediction)
        actual_values.append(row["rating"])

    predictions = np.array(
        predictions
    )

    actual_values = np.array(
        actual_values
    )

    # MAE
    mae = mean_absolute_error(
        actual_values,
        predictions
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predictions
        )
    )

    # Precision
    recommended = predictions >= 3.5
    relevant = actual_values >= 4.0

    if recommended.sum() > 0:

        precision = (
            (recommended & relevant).sum()
            /
            recommended.sum()
        )

    else:

        precision = 0

    results.append({
        "K": k,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Precision": round(
            precision,
            4
        )
    })


# ============================================================
# 10. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n========== RESULTS ==========")

print(results_df.to_string(index=False))


# ============================================================
# 11. FIND BEST K
# ============================================================

best_mae = results_df.loc[
    results_df["MAE"].idxmin()
]

best_rmse = results_df.loc[
    results_df["RMSE"].idxmin()
]

best_precision = results_df.loc[
    results_df["Precision"].idxmax()
]


print("\n========== BEST K ==========")

print(
    "Best K according to MAE:",
    int(best_mae["K"])
)

print(
    "Best K according to RMSE:",
    int(best_rmse["K"])
)

print(
    "Best K according to Precision:",
    int(best_precision["K"])
)


# ============================================================
# 12. FINAL RECOMMENDATION
# ============================================================

best_k = int(
    best_rmse["K"]
)

print(
    f"\nUsing K = {best_k} for final recommendations."
)


# ============================================================
# 13. COLLABORATIVE FILTERING RECOMMENDATION
# ============================================================

def cf_recommend(title, k=5):

    if title not in title_to_id:

        print(
            f"Movie '{title}' not found."
        )

        return pd.DataFrame()

    movie_id = title_to_id[title]

    if movie_id not in train_centered.index:

        print(
            f"'{title}' does not have enough "
            "training data."
        )

        return pd.DataFrame()

    number_of_neighbors = min(
        k + 1,
        len(train_centered)
    )

    model = NearestNeighbors(
        n_neighbors=number_of_neighbors,
        metric="cosine"
    )

    model.fit(
        train_centered.values
    )

    movie_position = (
        train_centered.index.get_loc(
            movie_id
        )
    )

    distances, indices = model.kneighbors(
        [train_centered.values[movie_position]]
    )

    recommendations = []

    for index, distance in zip(
        indices[0],
        distances[0]
    ):

        recommended_id = (
            train_centered.index[index]
        )

        if recommended_id == movie_id:
            continue

        recommendations.append({
            "title": id_to_title.get(
                recommended_id,
                "Unknown"
            ),
            "similarity": round(
                1 - distance,
                3
            )
        })

        if len(recommendations) == k:
            break

    return pd.DataFrame(
        recommendations
    )


print(
    "\n========== FINAL RECOMMENDATIONS =========="
)

print(
    cf_recommend(
        "Toy Story (1995)",
        k=best_k
    )
)


print("\n========== PROGRAM COMPLETE ==========")