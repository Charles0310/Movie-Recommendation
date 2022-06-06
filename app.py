# %%
# apikey 8265bd1679663a7ea12ac168da84d2e8
import pandas as pd
import streamlit as st
import pickle
import requests
from itertools import chain
import bz2
import _pickle as cPickle
# %%


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/" + \
        str(movie_id) + "?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "http://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


# %%
# smd = pickle.load(open('smd.pkl', 'rb'))
# cosine_sim = pickle.load(open('similarity.pkl', 'rb'))
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


smd = decompress_pickle('smd.pbz2')
cosine_sim = decompress_pickle('similarity.pbz2')

titles = smd['title'].values
genre = smd['genres'].values
genre = chain.from_iterable(genre)
genre = list(set(genre))
indices = pd.Series(smd.index, index=smd['title'])
titles = titles.tolist()
# %%


def weighted_rating(x):
    m = 434.0
    C = 5.244896612406511
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
# %%


def recommend(movie_title):
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movie_poster = []

    movies = smd.iloc[movie_indices][[
        'id', 'title', 'vote_count', 'vote_average', 'year']]
    id = movies.loc[movies['id'].notnull()]['id'].astype('int')
    vote_counts = movies.loc[movies['vote_count'].notnull(
    )]['vote_count'].astype('int')
    vote_averages = movies.loc[movies['vote_average'].notnull(
    )]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (
        movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified.vote_count.astype('int')
    qualified.id.astype('int')
    qualified.vote_average.astype('int')
    #qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('vote_average', ascending=False).head(9)
    for i in qualified['id'].values:
        movie_poster.append(fetch_poster(i))
    return qualified['title'].values, movie_poster
# %%
# rec =  recommend('Battleship')
# for i in rec:


#     print(i)
s = smd.apply(lambda x: pd.Series(x['genres']), axis=1).stack(
).reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = smd.drop('genres', axis=1).join(s)


def recommend_genre(genre):
    percentile = 0.85
    movie_poster = []
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()
                       ]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (
        df['vote_average'].notnull())][['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(lambda x: (
        x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(9)
    for i in qualified['id'].values:
        movie_poster.append(fetch_poster(i))
    return qualified['title'].values, movie_poster


movie_title = st.title("Movie recommendation")

movie_name = st.sidebar.selectbox(
    'Search Movies', titles
)
if st.sidebar.button('Recommend'):
    st.subheader('Recommended Movies')
    name, poster = recommend(movie_name)
#   print(movie_name)
    # for i in name, poster:
    #     st.write(i)
    col0, col1, col2 = st.columns(3)

    with col0:
        st.text(name[0])
        st.image(poster[0])
    with col1:
        st.text(name[1])
        st.image(poster[1])
    with col2:
        st.text(name[2])
        st.image(poster[2])

    col4, col5, col3 = st.columns(3)
    with col3:
        st.text(name[3])
        st.image(poster[3])
    with col4:
        st.text(name[4])
        st.image(poster[4])
    with col5:
        st.text(name[5])
        st.image(poster[5])

    col8, col6, col7 = st.columns(3)
    with col6:
        st.text(name[6])
        st.image(poster[6])
    with col7:
        st.text(name[7])
        st.image(poster[7])
    with col8:
        st.text(name[8])
        st.image(poster[8])


genre_name = st.sidebar.selectbox(
    'Search Genre', genre
)
if st.sidebar.button('Search'):
    st.subheader('Recommended Movies based on '+str(genre_name))
    name1, poster1 = recommend_genre(genre_name)
    col00, col01, col02 = st.columns(3)

    with col00:
        st.text(name1[0])
        st.image(poster1[0])
    with col01:
        st.text(name1[1])
        st.image(poster1[1])
    with col02:
        st.text(name1[2])
        st.image(poster1[2])

    col03, col04, col05 = st.columns(3)
    with col03:
        st.text(name1[3])
        st.image(poster1[3])
    with col04:
        st.text(name1[4])
        st.image(poster1[4])
    with col05:
        st.text(name1[5])
        st.image(poster1[5])
    col06, col07, col08 = st.columns(3)
    with col08:
        st.text(name1[8])
        st.image(poster1[8])
    with col06:
        st.text(name1[6])
        st.image(poster1[6])
    with col07:
        st.text(name1[7])
        st.image(poster1[7])
