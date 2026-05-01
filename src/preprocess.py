import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "E:/breast-cancer-img-classification/data/datasets"
OUTPUT_DIR = "E:/breast-cancer-img-classification/data/preprocessed_dataset"

CLASSES = ['0', '1']
SPLIT_RATIO = 0.8

def create_folders():
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def split_patients():
    patients = [
        p for p in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, p))
    ]

    random.shuffle(patients)

    split_idx = int(SPLIT_RATIO * len(patients))

    return patients[:split_idx], patients[split_idx:]

def process_patients(patient_list, split):
    for patient in patient_list:
        patient_path = os.path.join(SOURCE_DIR, patient)

        for cls in CLASSES:
            class_path = os.path.join(patient_path, cls)

            if not os.path.exists(class_path):
                continue

            for img in os.listdir(class_path):
                src = os.path.join(class_path, img)

                new_name = f"{patient}_{img}"
                dst = os.path.join(OUTPUT_DIR, split, cls, new_name)

                shutil.copy(src, dst)

def main():

    if os.path.exists(OUTPUT_DIR):
        print("Cleaning existing processed dataset...")
        shutil.rmtree(OUTPUT_DIR)
    
    print("Creating folders...")
    create_folders()

    print("Splitting patients...")
    train_patients, val_patients = split_patients()

    print("Processing training data...")
    process_patients(train_patients, "train")

    print("Processing validation data...")
    process_patients(val_patients, "val")

    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")

    print("Train cancer:", len(os.listdir(f"{train_dir}/1")))
    print("Train non-cancer:", len(os.listdir(f"{train_dir}/0")))

    print("Val cancer:", len(os.listdir(f"{val_dir}/1")))
    print("Val non-cancer:", len(os.listdir(f"{val_dir}/0")))

    print("Done.")

if __name__ == "__main__":
    main()