import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

df = pd.read_csv('./data/Test_data.tsv', sep='\t')  # Specify '\t' as the separator for TSV

#
valid_speakers = ["Chandler Bing","Monica Geller", "Ross Geller","Rachel Green","Joey Tribbiani","Phoebe Buffay"]
df_filtered = df[df['speaker'].isin(valid_speakers)]

# Step 2: Sort the DataFrame by the "speaker" column
df_sorted = df_filtered.sort_values(by='speaker')

# Step 3: (Optional) Save the sorted DataFrame back to a TSV file
df_sorted.to_csv('./data/sorted_test_data.tsv', sep='\t', index=False)
