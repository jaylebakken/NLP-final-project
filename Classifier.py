import math
from matplotlib import pyplot as plt
import pandas as pd
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch as tor
import scipy.sparse
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
class Friends_Classifier():


    def __init__(self):
        self.speakers = []
        current_speaker = "Chandler Bing"
        corpus = []
        df = pd.read_csv("./data/sorted_train_data.tsv", sep='\t')
        content = ""
        ## Parses through training data, creating one document for each of the six speakers
        for index, row in df.iterrows():
            if( row['speaker']==current_speaker):
                if(pd.notna(row['transcript'])):
                    content += row['transcript']
            else:
                corpus.append(content)
                content = ""
                self.speakers.append(current_speaker)
                current_speaker = row['speaker']
        corpus.append(content)
        self.speakers.append(current_speaker)
        ##^ makes sure that ross is added

        print(self.speakers)

        ##Creates TF-IDF matrix
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(corpus)
        print((self.X.shape))
        ## Feature names stored in instance variable for later use
        self.feat = self.vectorizer.get_feature_names_out()
        ## Constructs the SVD, with matrices U, S and VT
        self.U, self.S, self.VT = scipy.linalg.svd(self.X.toarray())


        # plot PCA embeddings
        pca = PCA(n_components=2)
        tfidf_pca = pca.fit_transform(self.X.toarray())
        df_pca = pd.DataFrame(tfidf_pca, columns=['PCA 1', 'PCA 2'])
        df_pca['Speaker'] = self.speakers
        plt.figure(figsize=(8, 6))
        plt.scatter(df_pca['PCA 1'], df_pca['PCA 2'], c='tab:blue', edgecolor='tab:orange', s=10)
        for i, txt in enumerate(df_pca['Speaker']):
            plt.annotate(txt, (df_pca['PCA 1'][i], df_pca['PCA 2'][i]),fontsize=12,color='tab:blue', ha='center', fontweight='bold')
        plt.title("Figure 1: 2D PCA of TF-IDF Matrix for the Friends Corpus")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.show()
       
        pass

    def What_friends_character_are_you(self, string_corpus):
        #tokenize input string
        user_corpus = string_corpus.split()
        #Construct query vector from user supplied query 
        tf = self.vectorizer.transform([string_corpus])

        #Perform least squares on every document in the right eigenspace
        alpha = ["Chandler Bing", "Joey Tribbiani", "Monica Geller", "Phoebe Buffay", "Rachel Green", "Ross Geller"]
        character_idx = 0
        least_square=10000000000
        boys = ["Chandler Bing","Joey Tribbiani","Ross Geller"]
        girls = ["Rachel Green","Monica Geller", "Phoebe Buffay"]
        boymatch=0
        girlmatch=0

        for r in range(len(alpha)):
            #find euclid. distance between two vectors
            dist = np.linalg.norm(np.transpose(tf.toarray()[0])-(self.VT[[r]][0]))
            #print("character = ",alpha[r]," score = ", dist)
            if (alpha[r] in boys):
                 boymatch+=dist
            else:
                 girlmatch+=dist

            if(abs(dist)<least_square):
                ## if inner product is lowest...
                least_square=abs(dist)
                character_idx=r
                ## update classification


        return (alpha[character_idx])
    
    

    

    
def main():
    fc = Friends_Classifier()
    df = pd.read_csv("./data/sorted_test_data.tsv", sep='\t')
    correct = 0
    total = 0
    boys = ["Chandler Bing","Joey Tribbiani","Ross Geller"]
    girls = ["Rachel Green","Monica Geller", "Phoebe Buffay"]
    for index, row in df.iterrows():
            if(row['transcript']!=[]):
                if( row['speaker'])==(fc.What_friends_character_are_you(row['transcript'])):
                    correct+=1
                total+=1
                print(total," <- row ->", fc.What_friends_character_are_you(row['transcript']))

    print("total accuracy... ", correct/total)


    speakers = []
    current_speaker = "Chandler Bing"
    corpus = []
    df = pd.read_csv("./data/sorted_train_data.tsv", sep='\t')
    content = ""
    for index, row in df.iterrows():
                if( row['speaker']==current_speaker):
                    if(pd.notna(row['transcript'])):
                        content += row['transcript']
                else:
                    corpus.append(content)
                    content = ""
                    speakers.append(current_speaker)
                    current_speaker = row['speaker']
    corpus.append(content)
    speakers.append(current_speaker)
        
    
    for friend in range(len(speakers)):
        print("running" , speakers[friend])
        print(fc.What_friends_character_are_you(corpus[friend]))
        if speakers[friend]==fc.What_friends_character_are_you(corpus[friend]):
            correct+=1
    print(correct)




if __name__ == "__main__":
   main()