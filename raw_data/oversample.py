import pandas as pd

def oversample(dataset, label, sample_size_frac=3.0):
    # Filter the dataframe to isolate the target rows
    if label == 'three_partition_label':
        target_rows = dataset[dataset[label] == 1]
    else:
        target_rows = dataset[dataset[label] > 3]
    
    # Sample from the filtered dataframe (keeping both columns)
    tmp = target_rows.sample(frac=sample_size_frac, replace=True)
    
    # Return the original dataset with the new duplicates appended
    return pd.concat([dataset, tmp], ignore_index=True)

def compare(df1,df2):
    malignant1 = df1[df1['three_partition_label']==1]
    malignant2 = df2[df2['three_partition_label']==1]


    fitz1 = df1[df1['fitzpatrick'] > 3]
    fitz2 = df2[df2['fitzpatrick'] > 3]

    message = f"DF1:\n\n Malignant: {malignant1.count().iloc[0]}\
                     Fitzpatrick: {fitz1.count().iloc[0]}\n\n \
                    Malignant ratio: {malignant1.count().iloc[0]/df1.count().iloc[0]}\
                    Fitzpatrick ratio: {fitz1.count().iloc[0]/df1.count().iloc[0]}\
                    \n\n DF2:\n\n\
                    Malignant: {malignant2.count().iloc[0]}\
                    Fitzpatrick: {fitz2.count().iloc[0]}\n\n \
                    Malignant ratio: {malignant2.count().iloc[0]/df2.count().iloc[0]}\
                    Fitzpatrick ratio: {fitz2.count().iloc[0]/df2.count().iloc[0]}"

    print(message)

df = pd.read_csv("Fitzpatrick17k_train_old.csv")
#df = df[['three_partition_label', 'fitzpatrick']]

# Map the text labels to 0 and 1
mapping = {
    'benign': 0,
    'non-neoplastic': 2,
    'malignant': 1
}

df['three_partition_label'] = df['three_partition_label'].map(mapping)

"""
count = df[df['three_partition_label'] == 1].count()

print("malignant:",count.iloc[0]/len(df),"%")

count_patrick = df[df['fitzpatrick'] == (5 or 6)].count()
print(f"patrick >5:{count_patrick}")

count_mal_patr = df[(df['fitzpatrick'] > 3) & (df['three_partition_label'] == 1)].count()

print(f"patrick < 3 and malignant: {count_mal_patr.iloc[0]}")
"""



malignant = df[df['three_partition_label'] == 1]

malignant_after = oversample(df,"three_partition_label",sample_size_frac=4.0)
print(malignant.count().iloc[0],malignant_after.count().iloc[0])

df_aug_mal = pd.concat([malignant_after,df])
df_mal = df_aug_mal[df_aug_mal['three_partition_label'] == 1]

print(f"Malignant: {df_mal}\n amlignant after: {df_aug_mal}\n malignant to benign ratio: {len(df_mal)/len(df_aug_mal)}")


fitz = df_aug_mal[df_aug_mal['fitzpatrick'] > 3]
fitz_after = oversample(fitz, "fitzpatrick",sample_size_frac=0.5)

df_aug_fitz = pd.concat([fitz_after,df_aug_mal],join='outer')
df_fitz = df_aug_fitz[df_aug_fitz['fitzpatrick'] > 3]

print(df_aug_fitz.count().iloc[0],df_fitz.count().iloc[0])

print(f"Fitzpatrick:\n {fitz.count().iloc[0]}\n fitz after:\n {fitz_after.count().iloc[0]}\n dark tone to light ratio: {len(df_fitz)/len(df_aug_fitz)}")


compare(df,df_aug_fitz)

print(df_aug_fitz)
reverse_mapping = {
    0: 'benign',
    2: 'non-neoplastic',
    1: 'malignant'
}
df_aug_fitz['three_partition_label'] = df_aug_fitz['three_partition_label'].map(reverse_mapping)
df_aug_fitz.to_csv("Fitzpatrick17k_train_augm.csv",index=False)