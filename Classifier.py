import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch as tor


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
        self.U, self.S, self.VT = np.linalg.svd(self.X.toarray(), full_matrices=False)

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

        ##Perform least squares on every document in the right eigenspace
        character_idx = 0
        least_square=10000000000
        for r in range(len(self.VT)):
            dist = np.dot(np.transpose(tf),self.VT[r])
            print("character = ",self.speakers[r]," score = ",dist)
            if(abs(dist)<least_square):
                character_idx=r
                least_square=abs(dist)


        
            
        
        return tuple([self.speakers[character_idx]])
    
def main():
    fc = Friends_Classifier()
    chandler=fc.What_friends_character_are_you("Finally, I figure I'd better answer it, and it turns out it's my mother, which is very-very weird, because- she never calls me!")
    print(chandler)

if __name__ == "__main__":
   main()