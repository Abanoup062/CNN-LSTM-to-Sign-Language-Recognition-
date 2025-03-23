# Sign Language Recognition using CNN-LSTM

## 📌 Overview
This project implements a **Sign Language Recognition** model using a **CNN-LSTM architecture**. It processes images of hand signs, extracts features using **MobileNetV2**, and classifies them using an **LSTM network**.

## 📂 Dataset
The dataset used for training and testing comes from the **Sign Language MNIST** dataset, which contains grayscale images of hand signs representing letters **A-Z**.

### 🔹 Data Structure
Each CSV file consists of:
- **Label**: The corresponding sign letter (0-25, representing A-Z)
- **Pixels**: 28x28 grayscale pixel values (784 features)

## 🛠 Installation
### 1️⃣ Clone the Repository
```bash
[git clone https://github.com/Abanoup062/CNN-LSTM-to-Sign-Language-Recognition-.git]
cd your-repo
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Model Architecture
The model consists of:
1️⃣ **Feature extraction** using **MobileNetV2** (pretrained on ImageNet)
2️⃣ **Reshaping** the extracted features for LSTM processing
3️⃣ **Bidirectional LSTM** for sequential feature analysis
4️⃣ **Dense layers** for classification

```python
model = Sequential([
    base_model,  # MobileNetV2 feature extractor
    Flatten(),
    Reshape((1, -1)),  # Reshape for LSTM
    Bidirectional(LSTM(256, return_sequences=False)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

## 📊 Data Visualization
### 🔹 Class Distribution
```python
sns.countplot(x=train_data.iloc[:, 0])
plt.title("Class Distribution in Training Data")
plt.show()
```
### 🔹 Sample Images
```python
def plot_sample_images(data, num_samples=10):
    images = data.iloc[:, 1:].values.reshape(-1, 28, 28)
    labels = data.iloc[:, 0].values
    indices = np.random.choice(len(data), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].set_title(f"Label: {labels[idx]}")
        axes[i].axis('off')
    plt.show()
```

## 🏋️ Training the Model
Run the script to train the model:
```bash
python train.py
```

## 📈 Performance Evaluation
The model is evaluated using:
- **Accuracy**
- **Confusion Matrix**
- **Loss & Accuracy Plots**

## 🤝 Contributing
Feel free to submit issues or pull requests!

## 📜 License
This project is licensed under the **MIT License**.

