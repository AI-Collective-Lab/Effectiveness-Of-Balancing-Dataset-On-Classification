# Effectiveness-Of-Balancing-Dataset-On-Classification
This Repository checks the effectiveness of balancing the dataset on Classification
## 📂 Project Structure

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
