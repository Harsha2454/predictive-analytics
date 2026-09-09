import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("shopping_history.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("Missing Values:")
print(df.isnull().sum())

df = df.dropna()

print("Available Users:")
print(df["UserID"].unique())

print("Available Products:")
print(df["Product"].unique())

user_product_matrix = df.pivot_table(
    index="UserID",
    columns="Product",
    values="Rating",
    aggfunc="mean",
    fill_value=0
)

print("User Product Matrix:")
print(user_product_matrix)

user_similarity = cosine_similarity(
    user_product_matrix
)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_product_matrix.index,
    columns=user_product_matrix.index
)

print("User Similarity:")
print(user_similarity_df)

selected_user = input(
    "Enter User ID: "
)

selected_interest = input(
    "Enter your interested category: "
)

user_data = df[
    df["UserID"] == selected_user
]

if user_data.empty:
    print("User not found")
else:

    user_index = list(
        user_product_matrix.index
    ).index(selected_user)

    similar_users = user_similarity_df[
        selected_user
    ].sort_values(
        ascending=False
    )

    similar_users = similar_users[
        similar_users.index != selected_user
    ]

    print("Most Similar Users:")

    print(similar_users.head(5))

    category_products = df[
        df["Category"].str.lower()
        == selected_interest.lower()
    ]

    purchased_products = set(
        user_data["Product"]
    )

    recommendations = category_products[
        ~category_products["Product"].isin(
            purchased_products
        )
    ]

    scores = {}

    for product in recommendations["Product"].unique():

        product_score = 0
        similarity_total = 0

        for similar_user, similarity in similar_users.head(10).items():

            user_ratings = df[
                (df["UserID"] == similar_user)
                &
                (df["Product"] == product)
            ]

            if not user_ratings.empty:

                rating = user_ratings["Rating"].mean()

                product_score += (
                    similarity * rating
                )

                similarity_total += abs(
                    similarity
                )

        if similarity_total > 0:

            scores[product] = (
                product_score /
                similarity_total
            )

    recommendation_df = pd.DataFrame(
        scores.items(),
        columns=[
            "Product",
            "Recommendation Score"
        ]
    )

    recommendation_df = recommendation_df.sort_values(
        "Recommendation Score",
        ascending=False
    )

    print("Recommended Products:")

    print(
        recommendation_df.head(10)
    )

    if recommendation_df.empty:

        print(
            "No recommendations available for this interest."
        )

    else:

        print(
            "Top Recommended Products:"
        )

        for product in recommendation_df.head(5)[
            "Product"
        ]:

            print(product)
