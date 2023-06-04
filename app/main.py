#########
# SETUP #
#########
# libraries 
import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.metrics.pairwise import cosine_similarity

# constants
DATA_PATH = "./data/"

#############
# LOAD DATA #
#############
df_anime = pd.read_csv(f"{DATA_PATH}anime.csv")
df_rating = pd.read_csv(f"{DATA_PATH}rating.csv")

#################
# DATA CLEANING #
#################
# replace -1 with null for missing ratings
df_rating['rating'] = df_rating['rating'].replace({-1: np.nan}, regex=True)

# keep 'TV' anime only
df_anime = df_anime[df_anime['type']=="TV"]

# merge data
df_merged = df_rating.merge(df_anime, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])
df_merged = df_merged.rename(columns={'rating_user': 'user_rating'})

# limit to users with id below 10,000 to save computing resources
df_merged = df_merged[['user_id', 'name', 'user_rating']]
df_merged_sub = df_merged[df_merged['user_id']<=10000]

# pivot on user_id
df_piv = df_merged_sub.pivot_table(index=['user_id'], columns=['name'], values='user_rating')

# normalize to standardize rating values
df_piv_norm = df_piv.apply(lambda x: (x-np.mean(x)) / (np.max(x)-np.min(x)), axis=1)

# drop all columns containing 0
df_piv_norm = df_piv_norm.fillna(0)
df_piv_norm = df_piv_norm.T
df_piv_norm = df_piv_norm.loc[:, (df_piv_norm!=0).any(axis=0)]

#########################
# RECOMMENDATION ENGINE #
#########################
# sparse matrix
piv_sparse = sp.sparse.csr_matrix(df_piv_norm.values)

# similarity 
item_similarity = cosine_similarity(piv_sparse)
# user_similarity = cosine_similarity(piv_sparse.T)

# insert similarity matrics into dataframes
df_item_sim = pd.DataFrame(item_similarity, index=df_piv_norm.index, columns=df_piv_norm.index)
# df_user_sim = pd.DataFrame(user_similarity, index=df_piv_norm.columns, columns=df_piv_norm.columns)

# top 10 anime recommendations
def get_anime_recommendations(anime_name):
    return df_item_sim.sort_values(by=anime_name, ascending=False).index[1:11]

###################
# WEB APPLICATION #
###################
def main():
    st.title("Anime Recommendation App")
    
    anime_name = st.text_input("Enter a anime name:")
    if st.button("Recommend"):
        recommendations = get_anime_recommendations(anime_name)

        st.subheader("Top 10 Recommended Animes:")
        for anime in recommendations:
            st.write(anime)

if __name__ == "__main__":
    main()

###################
# RUN APPLICATION #
###################
# streamlit run app/main.py