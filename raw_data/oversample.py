import pandas as pd

df = pd.read_csv("Fitzpatrick17k_full.csv")
df = df[['three_partition_label', 'fitzpatrick']]

# Map the text labels to 0 and 1
mapping = {
    'benign': 0,
    'non-neoplastic': 0,
    'malignant': 1
}

df['three_partition_label'] = df['three_partition_label'].map(mapping)
count = df[df['three_partition_label'] == 1].count()

print("malignant:",count.iloc[0]/len(df),"%")

count_patrick = df[df['fitzpatrick'] == (5 or 6)].count()
print(f"patrick >5:{count_patrick}")

count_mal_patr = df[(df['fitzpatrick'] > 3) & (df['three_partition_label'] == 1)].count()

print(f"patrick < 3 and malignant: {count_mal_patr.iloc[0]}")




malignant = df[df['three_partition_label'] == 1]
print(malignant.sample(500))

