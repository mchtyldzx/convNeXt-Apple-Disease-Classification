# 🍎 Apple Leaf Disease Detection

This project detects diseases on apple leaves taken from the [**Kaggle PlantVillage**](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data) dataset using the modern **ConvNeXt Small** architecture. In this study, the model performance is analyzed comparatively under two different training scenarios: **Transfer Learning (TL=True)** and **Fine-Tuning (TL=False)**.

##  Model Architecture: ConvNeXt-Small

The **ConvNeXt-Small** model used in this project is a modernized version of the CNN (Convolutional Neural Network) architecture with Vision Transformer (ViT) features. According to the analyses in the image, the prominent technical features of the model are as follows:

- **Patch-based Structure:** The image is processed by dividing it into small patches, which allows for better capture of local details.
- **Large Kernel Convolution:** Instead of standard 3x3, **7x7 kernel** sizes are used to provide a view of a wider area (receptive field).
- **LayerNorm:** **Layer Normalization** is preferred over classic Batch Normalization as it provides more stable training.
- **ImageNet Pretrained:** The model possesses weights pre-trained with the massive ImageNet dataset containing 1.3 million images.

### Transfer Learning Strategy

The model, which gained general object recognition capability on ImageNet, has been customized for apple leaves as follows:

- General feature extractors in the initial layers (edge, texture detection) are preserved.
- Only the final classification layer has been redesigned and trained according to apple diseases (4 classes) (**Fine-Tuning** in the sense of classifier training).

## 📂 Dataset Source: Kaggle PlantVillage

The data library used in the project is the open-source **PlantVillage** dataset, which is accepted as a standard worldwide. While the original dataset contains 38 different plant classes and more than 54,000 images, only data belonging to **Apple** leaves were filtered and used within the scope of this project.

### Apple Leaf Classes and Features Used

This dataset, consisting of photos taken in real field conditions, increases the model's success in the real world.

- 🍏 **Apple\_\_\_scab:** Scab disease
- 🍎 **Apple\_\_\_Black_rot:** Black rot
- 🍂 **Apple\_\_\_Cedar_apple_rust:** Cedar apple rust
- ✅ **Apple\_\_\_healthy:** Healthy leaves

### Data Distribution Table

The table below shows the distribution of the classes separated for the project into Training, Validation, and Test sets:

| Class                |  Train   |   Val   |  Test   | **Total** |
| :------------------- | :------: | :-----: | :-----: | :-------: |
| **Apple_scab**       |   445    |   95    |   90    |  **630**  |
| **Black_rot**        |   429    |   98    |   94    |  **621**  |
| **Cedar_apple_rust** |   183    |   47    |   45    |  **275**  |
| **Healthy**          |   1162   |   235   |   248   | **1645**  |
| **GRAND TOTAL**      | **2219** | **475** | **477** | **3171**  |

### ⚙️ Training Hyperparameters

The following fixed parameters were used for both scenarios throughout the project:

- **Number of Epochs (NUM_EPOCHS):** 50
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Weight Decay:** 0.0005

## 🏆 Test Results and Comparison (TL=True vs False)

Two different training strategies (Transfer Learning and Fine-Tuning) were compared within the scope of the project. The results obtained show that the **Transfer Learning (Feature Extraction)** method provides 100% success on this dataset.

## 📊 Training Graphs and Analysis

The training of the model was carried out on **Google Colab (NVIDIA Tesla T4 GPU)** hardware. During this time, the convergence behavior of the model is analyzed below.

### 1. Transfer Learning = True (Feature Extraction)

<img width="1136" height="451" alt="tltruegraph" src="https://github.com/user-attachments/assets/f7f1f221-0f7d-467d-8be4-26b6df52cc4b" />

- **Graph Behavior:** The Loss curve dropped very sharply to 0.1 levels within the first 5 epochs and then remained stable. The Accuracy graph climbed to the 98% level almost instantly and continued as a flat line.
- **Speed:** The model reached the convergence point extremely quickly.
- **Stability:** Training and Validation (Val) curves progressed very close to each other, showing no signs of overfitting.

### 2. Transfer Learning = False (Fine-Tuning)

<img width="1136" height="451" alt="tlfalsegraph" src="https://github.com/user-attachments/assets/cb1b77cb-5a0e-4463-9e38-2c9300a63200" />

- **Graph Behavior:** Loss values fell more slowly and in a fluctuating (zig-zag) manner. Significant fluctuations were observed in the Accuracy graph, especially during the first 20 epochs.
- **Stability:** The model had more difficulty learning the dataset and experienced instabilities while optimizing its weights from scratch.

> **CONCLUSION:** According to the graphical analyses; using Transfer Learning **increased the model's learning speed by approximately 2.5 times** and made the final accuracy more stable. Pre-trained (ImageNet) weights provide a critical performance boost on limited datasets.

## 🧩 Confusion Matrix Comparison

Detailed examination of model performance on the test set:

### TL = True Scenario

<img width="681" height="638" alt="tltrue" src="https://github.com/user-attachments/assets/dfb6b045-03db-4bff-a8d0-70583d1cf40b" />

When the confusion matrix is examined, it is seen that the model makes an **almost perfect** distinction.

- **Errors:** Almost non-existent. The number of correct predictions on the diagonal is maximum.
- **Example:** **All (88/88)** of the 88 examples in the `Apple_scab` class were classified correctly. There are no false positives or false negatives.

### TL = False Scenario

<img width="681" height="638" alt="tlfalse" src="https://github.com/user-attachments/assets/620ff176-ed8b-45cd-b2b3-e4303e11a7b5" />

Some confusion between classes stands out in the matrix.

- **Errors:** There are shifts especially between similar diseases.
- **Example:** Of the 88 examples in the `Apple_scab` class, **79** were correctly identified; however, **3** were incorrectly predicted as `Black_rot`, **1** as `Rust`, and **2** as `Healthy`.

### Scenario 1: Transfer Learning = True (Feature Extraction)

The main layers of the model were frozen, only the classifier was trained.
**Result: 100% Accuracy** 🥇

| Class                | Precision | Recall | F1-Score | Support |
| :------------------- | :-------: | :----: | :------: | :-----: |
| **Apple_scab**       |   1.00    |  1.00  |   1.00   |   88    |
| **Black_rot**        |   1.00    |  1.00  |   1.00   |   101   |
| **Cedar_apple_rust** |   1.00    |  1.00  |   1.00   |   45    |
| **Healthy**          |   1.00    |  1.00  |   1.00   |   243   |
| **ACCURACY**         |           |        | **1.00** | **477** |

### Scenario 2: Transfer Learning = False (Fine-Tuning)

All model layers were re-trained according to the dataset.
**Result: 98% Accuracy**

| Class                | Precision | Recall | F1-Score | Support |
| :------------------- | :-------: | :----: | :------: | :-----: |
| **Apple_scab**       |   1.00    |  0.93  |   0.96   |   85    |
| **Black_rot**        |   0.97    |  0.98  |   0.98   |   102   |
| **Cedar_apple_rust** |   0.98    |  1.00  |   0.99   |   44    |
| **Healthy**          |   0.98    |  1.00  |   0.99   |   246   |
| **ACCURACY**         |           |        | **0.98** | **477** |

> **Comment:** The fact that the `TL=True` scenario provides 100% success here proves that the ImageNet features of the ConvNeXt model are already very distinctive and sufficient for apple leaves. The attempt of the model to overfit to tiny variations in the dataset during fine-tuning caused the success to drop to 98%.

## 💻 Usage and Code

Project codes are available in the `convnext_small_apple.py` file. To run:

1. Install the necessary libraries with `pip install -r requirements.txt`.
2. Set the `DATA_DIR` path in the `convnext_small_apple.py` file.
3. Run with the command `python convnext_small_apple.py`.

> **Note:** The code runs with `transfer_learning = True` by default. If you want to perform Fine-Tuning, you can set the `transfer_learning` variable in the code to `False`.

---

**Model:** ConvNeXt Small | **Dataset:** PlantVillage > Apple Dataset | **Method:** Transfer Learning vs Fine-Tuning
