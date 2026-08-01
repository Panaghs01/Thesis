from zipfile import LZMA_VERSION
import torchvision.transforms as T
import pandas as pd
import os
from PIL import Image



### CONFIG ###

CSV_PATH = 'raw_data/Fitzpatrick17k_train.csv'          
IMAGE_DIR = 'raw_data/full_dataset'                          # Both original AND augmented images will live here
OUTPUT_CSV = 'raw_data/Fitzpatrick17k_balanced_train.csv'

# Column names in your CSV
PATH_COL = 'new_img_name'
LABEL_COL = 'three_partition_label' 
MINORITY_CLASS_LABEL = 'malignant'

augmentation = T.Compose([
    T.RandomVerticalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.9,1.1)),
    T.RandomRotation(degrees=180),
    T.ColorJitter()
])

def main():
    df = pd.read_csv(CSV_PATH)

    minority = df[df[LABEL_COL] == MINORITY_CLASS_LABEL]
    majority = df[df[LABEL_COL] != MINORITY_CLASS_LABEL]

    minority_count = len(minority)
    majority_count = len(majority)
    
    print(f"Majority class count: {majority_count}")
    print(f"Minority class count: {minority_count}")

    num_to_generate = majority_count - minority_count
    
    if num_to_generate <= 0:
        print("Dataset is already balanced or majority class is smaller. Exiting.")
        return

    print(f"Generating {num_to_generate} augmented images...")

    new_rows = []

    for i in range(num_to_generate):
        sample = minority.sample(1).iloc[0]

        base_name = os.path.basename(sample[PATH_COL])
        full_path = os.path.join(IMAGE_DIR, base_name)  

        try:
            img = Image.open(full_path).convert('RGB')

            aug_img = augmentation(img)

            name,ext = os.path.splitext(base_name)

            new_filename = f"{name}_augm_{i}{ext}"

            new_file_path = os.path.join(IMAGE_DIR,new_filename)

            aug_img.save(new_file_path)

            new_row = sample.copy()
            new_row[PATH_COL] = new_filename
            new_rows.append(new_row)

        except Exception as e:
            print(f"Error processing {full_path}:{e}")

    augmented_df = pd.DataFrame(new_rows)
    balanced_df = pd.concat([df,augmented_df])

    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    balanced_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Balancing complete! All images are located in: {IMAGE_DIR}")
    print(f"Saved balanced metadata to {OUTPUT_CSV}")
    print(f"New dataset size: {len(balanced_df)}")

if __name__ == "__main__":
    main()
