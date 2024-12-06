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

class Friends_Classifier():


    def __init__(self):
        self.speakers = []
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
                self.speakers.append(current_speaker)
                current_speaker = row['speaker']

        vectorizer = CountVectorizer()
        self.X = vectorizer.fit_transform(corpus)
        self.feat = vectorizer.get_feature_names_out()
        self.U, self.S, self.VT = scipy.linalg.svd(self.X.toarray(), full_matrices=False)
       
        pass

    def What_friends_character_are_you(self, string_corpus):
        user_corpus = string_corpus.split()
        #Construct query vector
        tf = [0]* len(self.feat)
        norm = 0
        for w in user_corpus:
            if w in self.feat:
                idx = np.where(self.feat==w) 
                tf[idx[0].item()]+=1

        
        print(self.VT.shape)
        #Perform least squares on every document in the right eigenspace
        friends=["chandler","joey","monica","rachel","pheobe","ross"]
        friends_idx=[101,308,420,504,486,524]
        character_idx = 0
        least_square=10000000000
        for r in range(len(friends)):
            dist = np.dot(np.transpose(tf),(self.VT[[friends_idx[r]]][0]))
            print("character = ",friends_idx[r]," score = ", dist)
            if(abs(dist)<least_square):
                least_square=abs(dist)
                character_idx=r
    
    
            
        
        return (friends[character_idx])
    
def main():
    fc = Friends_Classifier()
    chandler=fc.What_friends_character_are_you("")
    print(chandler)

if __name__ == "__main__":
   main()