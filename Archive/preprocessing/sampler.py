import pandas as pd

labels = ["label", "comment",  "author", "subreddit", "score", "ups", "downs", "date created", "utc", "parent comment", "id", "link_id"]

for chunk in pd.read_csv("sarc.csv", chunksize = 10000, sep = "\t", on_bad_lines = "skip", names = labels):
    with open("temp.csv", mode = 'w') as writer:
        chunk.to_csv("temp.csv", mode = 'w', sep = "|", index = False, header = True)
    break
