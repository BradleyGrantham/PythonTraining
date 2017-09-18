import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_columns', 50)

def ranker(df):
    df['rank'] = np.arange(len(df))+1
    return df

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

movie_ratings = pd.merge(movies, ratings)
full_data = pd.merge(users, movie_ratings)

most_rated = full_data.groupby(by='title').size().sort_values(ascending=False)[:25]
print most_rated
print full_data.title.value_counts()[:25]

highly_rated = full_data.groupby(by='title').agg({'rating': [np.size, np.mean]})
atleast100 = highly_rated['rating']['size'] > 100

print highly_rated[atleast100].sort_values(by=[('rating', 'mean')])[:25]

users.age.plot.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age');
plt.show()
