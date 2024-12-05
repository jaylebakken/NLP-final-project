import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

current_speaker = "#ALL#"
corpus = []
df = pd.read_csv("./data/sorted_file.tsv", sep='\t')
content = ""
for index, row in df.iterrows():
    if( row['speaker']==current_speaker):
        if(pd.notna(row['transcript'])):
            content += row['transcript']
    else:
        corpus.append(content)
        content = ""
        current_speaker = row['speaker']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
feat = vectorizer.get_feature_names_out()

U, S, VT = np.linalg.svd(X.toarray(), full_matrices=False)

print(VT)
print(feat)
