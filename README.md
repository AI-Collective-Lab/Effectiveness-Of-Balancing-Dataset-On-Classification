# Effectiveness-Of-Balancing-Dataset-On-Classification
This Repository checks the effectiveness of balancing the dataset on Classification
## 📂 Dataset
The dataset used in this project is hosted on Kaggle due to size constraints.

**[Download the Dataset Here](https://www.kaggle.com/datasets/muhammadazeemaif25/plant-disease-imbalanced-vs-gan-balanced)**

### Setup Instructions
1. Download the `plant_disease_gan_benchmark.zip` from the link above.
2. Unzip the file.
3. Move the folders (`combined`, `filtered`, `splitted`) into the `data/` directory of this repository.

Your final folder structure should look like this:
The dataset is organized into four main stages, representing the data preprocessing and augmentation pipeline:
```text
my-dataset-master/
│
├── Filtered_Plant_Dataset/          # Step 1: Quality Control
│   └── (Contains the raw leaf images after removing blurry or irrelevant samples)
│
├── Combined_Dataset/                # Step 2: Unification
│   └── (Merger of original splits. Created because the original source's 
│        test set had poor class distribution, requiring a fresh re-split)
│
├── Final_Split_Dataset/             # Step 3: Baseline Data
│   ├── train/                       # Imbalanced training data (The Baseline)
│   ├── val/
│   └── test/                        # A standard, well-distributed test set
│
└── Final_Split_Dataset+Generated/   # Step 4: The Solution (Balanced)
    ├── train/                       # Original Train + GAN-Generated Images
    ├── val/                         # (Same as above)
    └── test/                        # (Same as above - strictly real images)
```


---
