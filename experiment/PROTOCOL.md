# Evaluation protocol for the corrected re-run

Written **before** any model was trained, so the measures and the decision rule cannot be
chosen after seeing results. The code that computes everything below is in
[`metrics.py`](metrics.py).

## Why not accuracy
The 3 largest classes hold 29% of the images; the 7 smallest hold 4.4%. A model that got every
minority-class image wrong could still score about 95% accuracy, so accuracy says little about
whether balancing helped. It is reported for context only.

## Data
- Original PlantVillage colour images (54,305, 38 classes), one photo per leaf.
- Split 80 / 10 / 10 per class (stratified), with a fixed seed. Because each leaf appears once,
  no copy of a test leaf can be in training.
- **Minority classes** = the 7 classes with fewer than 500 images in the full dataset:
  Potato healthy (152), Apple cedar rust (275), Peach healthy (360), Raspberry healthy (371),
  Tomato mosaic virus (373), Grape healthy (423), Strawberry healthy (456).
- Synthetic (GAN) images go into the **training split only**. Validation and test are real images.
- The test split is used once per trained model, at the end. The epoch is chosen on validation.

## Methods compared
| Method | What changes |
|---|---|
| `baseline` | Train on the imbalanced training split as it is |
| `class_weighted` | Cross-entropy weighted by inverse class frequency |
| `oversampling` | Minority images repeated (with the usual augmentation) up to the target count |
| `gan_s1_cond_minority` | **S1:** one conditional DCGAN trained on the 7 minority classes only |
| `gan_s2_cond_all` | **S2:** one conditional DCGAN trained on all 38 classes (minority classes can borrow leaf structure from the large ones) |
| `gan_s3_per_class` | **S3:** seven plain DCGANs, one per minority class, each trained on that class alone |

**Target count:** every minority class is topped up to **874** training images, the median class size of
the training split. The same target is used for `oversampling` and all three GAN scenarios, so each
method adds the same 4,190 extra minority images and only *where they come from* differs:

| Class | Real (train) | Added | Share synthetic |
|---|--:|--:|--:|
| Potato healthy | 122 | 752 | 86% |
| Apple cedar rust | 220 | 654 | 75% |
| Peach healthy | 288 | 586 | 67% |
| Raspberry healthy | 297 | 577 | 66% |
| Tomato mosaic virus | 298 | 576 | 66% |
| Grape healthy | 338 | 536 | 61% |
| Strawberry healthy | 365 | 509 | 58% |

All GANs are trained on the **training split only**, at 64×64, the classifier's input size.
Same CNN, input size, augmentation, optimiser, epochs and splits for every method. 5 seeds each (see Design factors).

## Design factors (added 2026-10-04, before any balancing method was run)
Only one baseline run (seed 0, all real data, main CNN) had been seen when these were fixed.

**Imbalance level:** how many real training images each minority class keeps. Validation and test
are never changed.

| Level | Minority classes in the training split |
|---|---|
| `full` | all real images (122–365 per class) |
| `100` | 100 per class (Potato healthy keeps its 100 of 122) |
| `30` | 30 per class |

The reduced sets are nested (the 30 are a subset of the 100) and drawn once with a fixed seed, so
every method sees the same images. The top-up target stays at **874** per class at every level, so
the synthetic share grows as real data shrinks (97% at level 30). All GANs are retrained at each
level on that level's real images only.

**Model capacity:**

| Width | Model | Channels |
|---|---|---|
| `1.0` | `CropDiseaseCNN` (main) | 32-64-128-256, about 427k parameters |
| `0.25` | small version, same architecture | 8-16-32-64 |

**Seeds:** 5 per cell (0–4). Full grid: 6 methods × 3 levels × 2 widths × 5 seeds = 180 classifier
runs. GANs do not depend on the classifier, so both widths use the same generated images.

**Analysis:** the decision rule below is applied in every (level, width) cell. The main result is
the `CropDiseaseCNN` at each level; the small model shows whether the effect depends on capacity.
Every cell is reported, including those where no method helps.

## Measures
| Role | Measure |
|---|---|
| **Primary** | **Minority macro-F1**: mean F1 over the 7 minority classes |
| **Co-primary** | **Macro-F1** over all 38 classes |
| Secondary | Balanced accuracy (mean per-class recall) |
| Secondary | Per-class precision, recall and F1 for the minority classes and for every balanced class |
| Secondary | MCC (multiclass Matthews correlation) |
| Secondary | Macro PR-AUC (average precision) over the minority classes |
| Diagnostic | Confusion matrix |
| Context only | Overall accuracy |
| Not used | ROC-AUC (looks near-perfect on imbalanced one-vs-rest problems) |

Reported as **mean ± std over the 5 seeds**, with a **95% bootstrap CI** (1,000 resamples of the
test set) for the primary and co-primary measures. Training time per method is recorded too.

## GAN image quality
- **KID** (Kernel Inception Distance) per balanced class, generated vs real training images.
  KID is unbiased for small samples, unlike FID.
- **Train on synthetic, test on real**: a classifier trained only on generated images for a
  balanced class versus real images of the other classes, scored on real test images. Shows
  whether the generated images carry real disease features.

## Decision rule
A GAN scenario **works** only if its mean minority macro-F1 beats **each** of `baseline`,
`class_weighted` and `oversampling` by more than the larger seed standard deviation of the two
methods being compared (`gan_works()` in `metrics.py`). The rule is applied to S1, S2 and S3
separately. A scenario that only matches oversampling is reported as a negative result. The three
scenarios are then compared with each other on the same measures, to see which way of training
the GAN gives the most useful images.
