import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


    #Đọc dữ liệu, lấy 2 cột cần thiết
data = pd.read_csv("movie_data/movies.csv", encoding="latin-1", sep='\t', usecols=["title", "genres"])


data["genres"] = data["genres"].apply(lambda title: title.replace("|", " ").replace("-", ""))

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["genres"])
tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), index=data["title"],
                            columns=vectorizer.get_feature_names_out())
    #Dùng cosine_similarity để tính góc giữa 2 vector đặc trưng
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense = pd.DataFrame(cosine_sim, index=data["title"], columns=data["title"])

input_movie = "Sabrina (1995)"
top_k = 30
    #Lấy ra 30 bộ phim có khả năng giống thể loại nhất với phim Sabrina (1995)
result = cosine_sim_dense.loc[input_movie, :]
result = result.sort_values(ascending=False)[:top_k].to_frame("Score").reset_index()