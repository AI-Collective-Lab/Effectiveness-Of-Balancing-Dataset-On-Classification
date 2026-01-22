# Effectiveness-Of-Balancing-Dataset-On-Classification
This Repository checks the effectiveness of balancing the dataset on Classification
## 📂 Dataset
The dataset used in this project is hosted on Kaggle due to size constraints. 

**[Download the Dataset Here](https://www.kaggle.com/YOUR_USERNAME/YOUR_DATASET_NAME)**

### Setup Instructions
1. Download the `dataset.zip` from the link above.
2. Unzip the file.
3. Move the folders (`combined`, `filtered`, `splitted`) into the `data/` directory of this repository.

Your final folder structure should look like this:
```text
my-dataset-master/
│
├── combined/          # All images in one big folder (good for pre-training?)
│   ├── images/
│   └── labels.csv
│
├── filtered/          # The cleaned version (removed blurry/bad images)
│   ├── images/
│   └── labels.csv
│
└── splitted/          # The version ready for your ML pipeline
    ├── train/
    │   ├── class_A/
    │   └── class_B/
    ├── test/
    │   ├── class_A/
    │   └── class_B/
    └── val/
```


---
