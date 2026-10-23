# Machine Learning & Deep Learning Implementations 

A consolidated repository containing five end-to-end machine learning and deep learning projects developed on Kaggle for the **MATH 482** course. The collection covers the progression from domain-specific continuous regression to advanced feature engineering on noisy synthetic spaces, multi-layer perceptrons, and convolutional neural networks.

---

## Table of Contents

- [Overview & Objectives](#overview--objectives)
- [Technologies & Tooling](#technologies--tooling)
- [Projects Index](#projects-index)
- [Detailed Project Breakdown](#detailed-project-breakdown)
  - [Project 1: Wild Blueberry Yield Prediction](#project-1-wild-blueberry-yield-prediction)
  - [Project 2: Abstract Multi-Class Classification & Noise Filtering](#project-2-abstract-multi-class-classification--noise-filtering)
  - [Project 3: Mixed-Type Abstract Regression & Algorithm Comparison](#project-3-mixed-type-abstract-regression--algorithm-comparison)
  - [Project 4: Deep Tabular Classification & Hyperparameter Optimization](#project-4-deep-tabular-classification--hyperparameter-optimization)
  - [Project 5: Geometric Shape Classification with CNNs](#project-5-geometric-shape-classification-with-cnns)

---

## Overview & Objectives

This repository unifies five computational projects designed to simulate practical data science challenges. In applied environments, data scientists frequently encounter abstract, noisy, or uncurated datasets where domain intuition is absent. 

The primary goals demonstrated across this work include:
- Modeling continuous real-world phenomena with biological and meteorological predictors.
- Formulating statistical heuristics to detect and strip irrelevant random noise columns and redundant collinear features.
- Managing mixed-attribute tabular spaces with high missingness, outlier leverage, and categorical variations (both ordinal and nominal).
- Architectural tuning and regularization of deep feedforward neural networks (MLPs) against complex decision surfaces.
- Training computer vision architectures (CNNs) capable of classifying synthetic shapes across varying orientations, background artifacts, and occluding noise.

--- 

## Technologies & Tooling

- **Languages & Environments:** Python 3.9+, Jupyter Notebooks, Kaggle Kernels
- **Data Manipulation & Scientific Computing:** NumPy, Pandas
- **Exploratory Data Analysis & Visualization:** Matplotlib, Seaborn
- **Classical Machine Learning:** Scikit-Learn (Linear Models, Polynomial Features, Support Vector Machines, Preprocessing, Metrics)
- **Deep Learning Frameworks:** PyTorch / TensorFlow / Keras (Multi-Layer Perceptrons, Convolutional Neural Networks)
- **Computer Vision & Augmentation:** OpenCV, PIL, Torchvision / Keras Image Preprocessing

---

## Projects Index

| # | Project | Task / Domain | Key Challenges & Focus | Primary Methods & Models | Directory |
|:---:|:---|:---|:---|:---|:---:|
| **01** | **Wild Blueberry Yield Prediction** | Tabular Regression (Agricultural) | Multicollinear weather variables, non-linear bee population dynamics | Correlation pruning, feature scaling, continuous linear and ensemble regression | [`01-blueberry-yield-prediction/`](https://github.com/aysgard/Machine-Learning-Algorithms/blob/main/01-blueberry-yield-prediction.ipynb) |
| **02** | **Abstract Multi-Class Classification** | 4-Class Classification (Synthetic) | 27 abstract features with random noise columns, redundant linear combinations, and missing values | Mutual information, VIF collinearity pruning, missing value imputation, multi-class model benchmarking | [`02-multiclass-noise-filtering/`](https://github.com/aysgard/Machine-Learning-Algorithms/blob/main/02-multiclass-noise-filtering.ipynb) |
| **03** | **Mixed-Type Abstract Regression** | Continuous Regression (Synthetic) | Complex target interactions, mixed types (continuous, ranked ordinal, nominal), high missingness | Custom ordinal rank mapping, one-hot nominal encoding, Linear Regression, Polynomial Regression, SVR, MLP | [`03-mixed-type-regression/`](https://github.com/aysgard/Machine-Learning-Algorithms/blob/main/03-mixed-type-regression.ipynb) |
| **04** | **Deep Tabular Classification** | Complex Non-Linear Classification (Synthetic) | High non-linearity, dirty data, significant overfitting risks | Deep MLP architectures, activation comparisons (ReLU, LeakyReLU), optimizer tuning (SGD, Adam, RMSprop), Dropout, L2 decay | [`04-deep-tabular-classification/`](https://github.com/aysgard/Machine-Learning-Algorithms/blob/main/04-deep-tabular-classification.ipynb) |
| **05** | **Geometric Shape Classification** | Computer Vision (128×128 Image Classification) | 6 geometric shapes with random rotation, scale, occluding line artifacts, and diverse background noise types | Dynamic data augmentation, Conv2D, MaxPooling, Batch Normalization, learning rate schedules, Softmax CNN | [`05-shape-image-cnn/`](https://github.com/aysgard/Machine-Learning-Algorithms/blob/main/05-shape-image-cnn.ipynb) |

---

## Detailed Project Breakdown

### Project 1: Wild Blueberry Yield Prediction

* **Task & Goal:** Predict commercial blueberry yield values from environmental, physical, and pollinator characteristics.
* **Domain Context:** Blueberries are perennial flowering plants (*Cyanococcus*, genus *Vaccinium*). In commercial production, lowbush (wild) and highbush varieties depend heavily on spatial clone size, multiple pollinator species densities, and balanced temperatures during the bloom season.
* **Feature Schema:**
  * `Clonesize` ($m^2$): Average blueberry clone size in the field.
  * Pollinator Densities ($bees/m^2/min$): `Honeybee`, `Bumbles`, `Andrena`, and `Osmia`.
  * Temperature Extremes & Means ($^\circ C$): Upper band records (`MaxOfUpperTRange`, `MinOfUpperTRange`, `AverageOfUpperTRange`) and lower band records (`MaxOfLowerTRange`, `MinOfLowerTRange`, `AverageOfLowerTRange`).
  * Precipitation Metrics ($Days$): `RainingDays` and `AverageRainingDays` during the bloom window.
* **Methodology:**
  * Assessed multicollinearity across upper and lower daily temperature bands.
  * Standardized continuous physical and weather predictors.
  * Evaluated baseline and regularized continuous regression models.

---

### Project 2: Abstract Multi-Class Classification & Noise Filtering

* **Task & Goal:** Predict four discrete class labels (`0`, `1`, `2`, `3`) on an uncurated synthetic dataset containing 35,000 training observations and 15,000 test observations.
* **Domain Context:** Simulates working with blind, proprietary, or domain-abstract data where domain expertise cannot guide feature engineering.
* **Core Engineering Challenges:**
  * **Irrelevant Column Identification:** Identifying randomly generated noise features that carry zero statistical dependence on `target`.
  * **Redundancy Pruning:** Identifying features that are exact or near-exact linear combinations of other columns.
  * **Data Quality:** Systematic imputation of null values and treatment of severe outlier points.
* **Methodology:**
  * Filtered variables using correlation thresholds and Variance Inflation Factor (VIF).
  * Evaluated statistical dependence using Mutual Information classifiers.
  * Implemented an internal validation split and benchmarked two distinct classification algorithms on both training and validation sets.

---

### Project 3: Mixed-Type Abstract Regression & Algorithm Comparison

* **Task & Goal:** Estimate a continuous target variable from a 21-feature synthetic dataset featuring complex non-linear relationships, 30,000 training rows, and 10,000 test rows.
* **Feature Typology:**
  * **Continuous:** Real-valued numerical floats (e.g., `-324.39`, `0.22`).
  * **Ordinal:** Inherently ranked alphanumeric codes (e.g., `A1` < `A2` < `A3`).
  * **Nominal:** Unordered alphanumeric labels (e.g., `xx`, `xy`, `p`, `q`).
* **Core Engineering Challenges:**
  * Identifying and dropping randomly generated noise features.
  * Preserving natural mathematical rank in ordinal features while applying one-hot/frequency encoding to nominal features.
  * Managing elevated missingness rates and extreme outlier points.
* **Algorithm Benchmark:**
  1. *Linear Regression* (Parametric baseline)
  2. *Polynomial Regression* (Non-linear polynomial expansion)
  3. *Support Vector Regression (SVR)* (Kernel-based optimization)
  4. *Feedforward Neural Network / MLP* (Multi-layer representation learning)

---

### Project 4: Deep Tabular Classification & Hyperparameter Optimization

* **Task & Goal:** Classify target classes from 22,500 training points and 7,500 test points across 21 features with highly non-linear decision boundaries using Deep Learning.
* **Domain Context:** Where linear and simple tree baselines degrade due to high structural complexity and noise, multi-layer neural networks can learn hierarchical representations if regularized properly.
* **Engineering & Model Pipeline:**
  * Automated data preprocessing pipeline handling irrelevant column pruning, mixed encoding, and outlier scaling.
  * Systematic search over MLP hyperparameters:
    * **Depth & Breadth:** Varying layer count and hidden unit distribution.
    * **Activation Functions:** Comparing convergence across ReLU, LeakyReLU, and alternative activations.
    * **Optimization Algorithms:** Comparing momentum-accelerated SGD, Adam, and RMSprop.
    * **Regularization Strategies:** Applying Dropout masks and weight decay (L2 penalty) to constrain training vs. validation loss divergence.

---

### Project 5: Geometric Shape Classification with CNNs

* **Task & Goal:** Classify synthetic $128 \times 128$ pixel images into six distinct geometric classes:
  * `0`: Circle
  * `1`: Triangle
  * `2`: Rectangle
  * `3`: Pentagon
  * `4`: Hexagon
  * `5`: Star
* **Core Computer Vision Challenges:**
  * Objects appear with arbitrary scales, translations, and rotation angles.
  * Random background environments: solid colors, pure white, pure black, uniform noise, and blurred noise patterns.
  * Occlusion artifacts introduced via randomly placed line strokes across the visual field.
* **Deep Learning Architecture & Augmentation:**
  * Pipeline utilizing standard image normalization ($[0, 1]$ or $[-1, 1]$ scaling).
  * Data augmentation strategies including random affine transformations, slight rotations, and color jitter.
  * Deep Convolutional Neural Network (Conv2D $\rightarrow$ BatchNormalization $\rightarrow$ Non-linearity $\rightarrow$ Pooling $\rightarrow$ Dense Classification Head with Softmax).
  * Regularized using Dropout, weight decay, and learning rate scheduling.

