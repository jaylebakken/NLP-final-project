from sklearn.feature_extraction.text import CountVectorizer


class TermDocument():


    def init():
        cv = CountVectorizer
        path = "./data/{}.txt"
        with open(path.format("neg")) as neg_data_f:
            neg_data = neg_data_f.read().split()

        X = cv.fit_transform(neg_data)

        # Get the feature names (i.e., the terms)
        feature_names = cv.get_feature_names_out()

        # Create a DataFrame from the DTM
        import pandas as pd
        df = pd.DataFrame(X.toarray(), columns=feature_names)

        print(df)
        