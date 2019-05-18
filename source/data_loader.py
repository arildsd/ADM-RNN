import pandas

FILEPATH = "../data/amazon_reviews.csv"

df = pandas.read_csv(FILEPATH)
df = df[["Score", "Text"]]
print(df.head())