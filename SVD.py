import pandas as pd

# Read the TSV file into a DataFrame
df = pd.read_csv("./data/friends_transcripts.tsv", sep="\t")

# Sort the DataFrame by the specified column
sorted_df = df.sort_values(by="speaker")

# Write the sorted DataFrame back to a TSV file (optional)
sorted_df.to_csv("sorted_file.tsv", sep="\t", index=False)