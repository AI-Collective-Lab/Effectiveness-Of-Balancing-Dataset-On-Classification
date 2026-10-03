# Effectiveness-Of-Balancing-Dataset-On-Classification
This Repository checks the effectiveness of balancing the dataset for classification
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


## 📊 Class distribution (original PlantVillage)

The original **PlantVillage** colour images (one photo per leaf, before any augmentation) are **heavily imbalanced**. This is the version used for the corrected re-run of this project.

- **38 classes, 54,305 images**: 1,429 per class on average
- Largest class **5,507** vs smallest **152**: an imbalance ratio of **36.2 : 1**
- **7 classes have fewer than 500 images**; the 3 largest classes hold **29%** of all images
- The Kaggle *“New Plant Diseases Dataset (Augmented)”* used in the first experiments had been artificially rebalanced with rotated and flipped copies (max/min only 1.8 : 1), which hid this imbalance.

Source: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) · Kaggle mirror: [abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (`color/` folder)

| # | Plant | Class | Images | Share | Distribution |
|--:|---|---|--:|--:|---|
| 1 | Orange | Huanglongbing (Citrus greening) | 5,507 | 10.1% | `████████████████████` |
| 2 | Tomato | Tomato Yellow Leaf Curl Virus | 5,357 | 9.9% | `███████████████████` |
| 3 | Soybean | Healthy | 5,090 | 9.4% | `██████████████████` |
| 4 | Peach | Bacterial spot | 2,297 | 4.2% | `████████` |
| 5 | Tomato | Bacterial spot | 2,127 | 3.9% | `████████` |
| 6 | Tomato | Late blight | 1,909 | 3.5% | `███████` |
| 7 | Squash | Powdery mildew | 1,835 | 3.4% | `███████` |
| 8 | Tomato | Septoria leaf spot | 1,771 | 3.3% | `██████` |
| 9 | Tomato | Spider mites Two-spotted spider mite | 1,676 | 3.1% | `██████` |
| 10 | Apple | Healthy | 1,645 | 3.0% | `██████` |
| 11 | Tomato | Healthy | 1,591 | 2.9% | `██████` |
| 12 | Blueberry | Healthy | 1,502 | 2.8% | `█████` |
| 13 | Bell pepper | Healthy | 1,478 | 2.7% | `█████` |
| 14 | Tomato | Target Spot | 1,404 | 2.6% | `█████` |
| 15 | Grape | Esca (Black Measles) | 1,383 | 2.5% | `█████` |
| 16 | Corn (maize) | Common rust | 1,192 | 2.2% | `████` |
| 17 | Grape | Black rot | 1,180 | 2.2% | `████` |
| 18 | Corn (maize) | Healthy | 1,162 | 2.1% | `████` |
| 19 | Strawberry | Leaf scorch | 1,109 | 2.0% | `████` |
| 20 | Grape | Leaf blight (Isariopsis Leaf Spot) | 1,076 | 2.0% | `████` |
| 21 | Cherry (incl. sour) | Powdery mildew | 1,052 | 1.9% | `████` |
| 22 | Potato | Early blight | 1,000 | 1.8% | `████` |
| 23 | Potato | Late blight | 1,000 | 1.8% | `████` |
| 24 | Tomato | Early blight | 1,000 | 1.8% | `████` |
| 25 | Bell pepper | Bacterial spot | 997 | 1.8% | `████` |
| 26 | Corn (maize) | Northern Leaf Blight | 985 | 1.8% | `████` |
| 27 | Tomato | Leaf Mold | 952 | 1.8% | `███` |
| 28 | Cherry (incl. sour) | Healthy | 854 | 1.6% | `███` |
| 29 | Apple | Apple scab | 630 | 1.2% | `██` |
| 30 | Apple | Black rot | 621 | 1.1% | `██` |
| 31 | Corn (maize) | Cercospora / Gray leaf spot | 513 | 0.9% | `██` |
| 32 | Strawberry | Healthy ⚠️ | 456 | 0.8% | `██` |
| 33 | Grape | Healthy ⚠️ | 423 | 0.8% | `██` |
| 34 | Tomato | Tomato mosaic virus ⚠️ | 373 | 0.7% | `█` |
| 35 | Raspberry | Healthy ⚠️ | 371 | 0.7% | `█` |
| 36 | Peach | Healthy ⚠️ | 360 | 0.7% | `█` |
| 37 | Apple | Cedar apple rust ⚠️ | 275 | 0.5% | `█` |
| 38 | Potato | Healthy ⚠️ | 152 | 0.3% | `█` |

⚠️ = fewer than 500 images (the minority classes). Bars are scaled to the largest class (5,507 images).


## 📏 Evaluation protocol (corrected re-run)

The measures and the decision rule were fixed **before** re-running, in [`experiment/PROTOCOL.md`](experiment/PROTOCOL.md); the code is [`experiment/metrics.py`](experiment/metrics.py).

| Role | Measure |
|---|---|
| **Primary** | **Minority macro-F1**: mean F1 over the 7 classes with fewer than 500 images |
| **Co-primary** | **Macro-F1** over all 38 classes |
| Secondary | Balanced accuracy · MCC · minority macro PR-AUC · per-class precision / recall / F1 |
| Context only | Overall accuracy |

**Why not accuracy:** the 7 minority classes are only 4.4% of the images. A model that gets every one of them wrong still scores **95.6% accuracy**, but **0.0 minority macro-F1**.

Four methods are compared on identical splits, 3 seeds each, scored on an untouched test set: no balancing, class-weighted loss, oversampling, and GAN images (training split only). GAN balancing counts as working only if it beats **all three** by more than the seed-to-seed variation. Generated images are also scored with KID and a train-on-synthetic, test-on-real check.

---

## 👥 Contributors

| | Name | Role |
|---|---|---|
| <img src="https://github.com/Muhammad-Azeem-Bhatti.png?size=60" width="40" height="40"> | [@Muhammad-Azeem-Bhatti](https://github.com/Muhammad-Azeem-Bhatti) | Author |
| <img src="https://github.com/munahilamin03.png?size=60" width="40" height="40"> | [@munahilamin03](https://github.com/munahilamin03) | Co-author |
| <img src="https://github.com/zainulaabaidin.png?size=60" width="40" height="40"> | [@zainulaabaidin](https://github.com/zainulaabaidin) | Co-author |
| <img src="https://github.com/Sumer238.png?size=60" width="40" height="40"> | [@Sumer238](https://github.com/Sumer238) (Sumer Iqbal) | Contributor |
