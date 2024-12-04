from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
corpus = []
content = ""
with open("./data/neg.txt", 'r') as file:
    for line in file:
        for w in line:
           content+=w
        corpus.append(content)
        content = ""
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()

U, S, VT = np.linalg.svd(X.toarray(), full_matrices=False)

print(VT)

