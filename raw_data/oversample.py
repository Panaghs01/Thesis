import pandas as pd


def oversample(dataset, label, sample_size_frac=3.0):
    """
    Return only the sampled duplicates of rows matching the given label.
    For 'three_partition_label', target rows are those labeled 1.
    Otherwise, target rows are those where the label column value > 3.
    """
    if label == 'three_partition_label':
        target_rows = dataset[dataset[label] == 1]
    else:
        target_rows = dataset[dataset[label] > 3]
    # Sample with replacement; return only the duplicates
    return target_rows.sample(frac=sample_size_frac, replace=True)


def compare(df1, df2):
    malignant1 = df1[df1['three_partition_label'] == 1]
    malignant2 = df2[df2['three_partition_label'] == 1]

    fitz1 = df1[df1['fitzpatrick'] > 3]
    fitz2 = df2[df2['fitzpatrick'] > 3]

    message = (
        f"DF1:\n\n"
        f" Malignant: {malignant1.count().iloc[0]}  "
        f"Fitzpatrick: {fitz1.count().iloc[0]}\n\n"
        f" Malignant ratio: {malignant1.count().iloc[0]/df1.count().iloc[0]}  "
        f"Fitzpatrick ratio: {fitz1.count().iloc[0]/df1.count().iloc[0]}\n\n"
        f"DF2:\n\n"
        f" Malignant: {malignant2.count().iloc[0]}  "
        f"Fitzpatrick: {fitz2.count().iloc[0]}\n\n"
        f" Malignant ratio: {malignant2.count().iloc[0]/df2.count().iloc[0]}  "
        f"Fitzpatrick ratio: {fitz2.count().iloc[0]/df2.count().iloc[0]}"
    )

    print(message)


def create_csv(path, frac_fitz=1.0, frac_mal=1.0):
    # Read the dataset specified by the caller
    df = pd.read_csv(path)

    # Map the text labels to numeric values
    mapping = {
        'benign': 0,
        'non-neoplastic': 2,
        'malignant': 1
    }
    df['three_partition_label'] = df['three_partition_label'].map(mapping)

    # Oversample malignant class
    malignant_duplicates = oversample(df, "three_partition_label", sample_size_frac=frac_mal)
    df_aug_mal = pd.concat([df, malignant_duplicates], ignore_index=True)
    print(f"Malignant rows before: {df['three_partition_label'].eq(1).sum()}, after: {df_aug_mal['three_partition_label'].eq(1).sum()}")

    # Oversample Fitzpatrick > 3 class
    fitz = df_aug_mal[df_aug_mal['fitzpatrick'] > 3]
    fitz_duplicates = oversample(fitz, "fitzpatrick", sample_size_frac=frac_fitz)
    df_aug_fitz = pd.concat([df_aug_mal, fitz_duplicates], ignore_index=True)
    print(f"Fitzpatrick >3 rows before: {fitz.shape[0]}, after: {df_aug_fitz[df_aug_fitz['fitzpatrick'] > 3].shape[0]}")

    compare(df, df_aug_fitz)

    # Convert numeric labels back to text for output
    reverse_mapping = {
        0: 'benign',
        2: 'non-neoplastic',
        1: 'malignant'
    }
    df_aug_fitz['three_partition_label'] = df_aug_fitz['three_partition_label'].map(reverse_mapping)

    print(df_aug_fitz['three_partition_label'].value_counts())
    df_aug_fitz.to_csv(path + "_aug", index=False)


path = 'Fitzpatrick17k_val.csv'
create_csv(path, frac_fitz=1.0, frac_mal=3.0)
