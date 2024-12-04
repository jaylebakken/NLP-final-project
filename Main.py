from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer
path = "./data/{}.txt"
negdata = []
content = ""
with open("./data/neg.txt", 'r') as file:
    for line in file:
        for w in line:
           content+=w
        negdata.append(content)
        content = ""
       
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

X = cv.fit_transform(corpus)

# Get the feature names (i.e., the terms)
feature_names = cv.get_feature_names_out()
