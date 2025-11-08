# **Facial Emotion Recognition**

<div align="center">

![Status](https://img.shields.io/badge/status-In%20Development-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

A deep learning-based system for detecting human emotions from facial expressions using Convolutional Neural Networks (CNN).

</div>

## 📋 **Overview**

This project implements an emotion recognition system that can classify facial expressions into 7 different emotions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

The model is trained on the `FER2013 dataset` and achieves approximately `66% accuracy` on the test set.

> **⚠️ Note**: This is a work-in-progress project. Future versions will include:
> - Higher accuracy models (transfer learning with pre-trained networks)
> - Real-time emotion detection using webcam
> - Enhanced preprocessing techniques
> - Model optimization for edge devices
> - Web-based demo interface


## ✨ **Features**

- **Custom CNN Architecture**: Deep convolutional neural network with batch normalization and dropout
- **Data Augmentation**: Comprehensive augmentation pipeline to improve model generalization
- **Class Balancing**: Automatic class weight calculation to handle imbalanced dataset
- **Training Monitoring**: Callbacks for early stopping, learning rate reduction, and model checkpointing
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrix, and misclassification analysis


## 🚀 **Installation**

### Prerequisites:
- Python 3.8 or higher
- pip package manager
- Git LFS (for model files)

### Setup:

1. **Clone the repository**
```bash
git clone https://github.com/BraveVahid/facial_emotion_recognition.git
cd facial_emotion_recognition
```

2. **Install Git LFS** (if not already installed)
```bash
# On Ubuntu/Debian
sudo apt-get install git-lfs

# On macOS
brew install git-lfs

# Initialize Git LFS
git lfs install
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download the FER2013 dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Extract and place in the `data/` directory with the following structure:
```
data/
└── fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```


## **💻 Usage**

### Training the Model:
```bash
python -m src.train
```

Training parameters (can be modified in `src/train.py`):
- batch size: 64
- epochs: 100
- Automatic early stopping with patience of 15 epochs
- Learning rate reduction on plateau

### Evaluating the Model

Open and run the `reports.ipynb` notebook to:
- Load the trained model
- Evaluate on test dataset
- Generate classification report
- Visualize confusion matrix
- Analyze misclassified samples


## **🏗️ Model Architecture**

The model consists of:

**Convolutional Blocks:**
- 4 convolutional blocks with increasing filter sizes (64 → 128 → 256 → 512)
- Batch normalization after each convolution
- ReLU activation
- Max pooling (2×2)
- Dropout layers (0.25 → 0.3 → 0.4 → 0.5)

**Fully Connected Layers:**
- Flatten layer
- Dense layer (1024 neurons) + BatchNorm + Dropout (0.5)
- Dense layer (512 neurons) + BatchNorm + Dropout (0.5)
- Output layer (7 neurons, softmax activation)

**Regularization:**
- L2 regularization (0.0001) on all layers
- Dropout at multiple stages
- Data augmentation during training

**Total Parameters:** ~10.5M


## **📁 Project Structure**
```
emotion-detection/
├── .gitattributes          # Git LFS configuration
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── reports.ipynb          # Model evaluation notebook
├── data/                  # Dataset directory (not tracked)
│   └── fer2013/
│       ├── train/
│       └── test/
├── model/                 # Saved models and training history
│   ├── emotion_recognition.keras
│   ├── training_history.json
│   ├── training_history.csv
│   └── training_history.pkl
└── src/                   # Source code
    ├── data_loader.py     # Data loading and augmentation
    ├── model.py           # Model architecture
    └── train.py           # Training script
```

## **📊 Results:**

### Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 66.13% |
| Test Loss | 1.1089 |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.55 | 0.64 | 0.59 | 958 |
| Disgust | 0.55 | 0.67 | 0.60 | 111 |
| Fear | 0.56 | 0.35 | 0.43 | 1024 |
| Happy | 0.92 | 0.83 | 0.87 | 1774 |
| Neutral | 0.56 | 0.74 | 0.64 | 1233 |
| Sad | 0.56 | 0.51 | 0.53 | 1247 |
| Surprise | 0.72 | 0.84 | 0.78 | 831 |

### Key Observations:

- **Best Performance**: Happy emotion (92% precision, 87% F1-score)
- **Challenging Classes**: Fear (43% F1-score) and Sad (53% F1-score)
- **Balanced Accuracy**: Model shows relatively consistent performance across most classes
- The confusion matrix reveals common misclassifications between similar emotions

## **🔮 Future Work**

This project is actively under development. Planned improvements include:

### Short-term Goals
- [ ] Implement real-time webcam emotion detection
- [ ] Add data preprocessing improvements
- [ ] Create a web-based demo interface
- [ ] Optimize model for faster inference

### Medium-term Goals
- [ ] Experiment with transfer learning (VGG, ResNet, EfficientNet)
- [ ] Implement ensemble methods for better accuracy
- [ ] Add multi-face detection support
- [ ] Create REST API for model deployment

### Long-term Goals
- [ ] Support for video emotion tracking


## **👨‍💻 Author**

#### Vahid Siyami
- GitHub: [@BraveVahid](https://github.com/BraveVahid)
- Email: vahidsiyami.dev@gmail.com
- Telegram: [@BraveVahid](https://t.me/BraveVahid)

## **🤝 Contributing**

Contributions are welcome and greatly appreciated!

**1. Fork the Repository**

Click the "Fork" button at the top right of this repository.

**2. Clone Your Fork**
```bash
git clone https://github.com/your-username/facial_emotion_recognition.git
cd facial_emotion_recognition
```

**3. Create a Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**4. Make Your Changes**
- Write clean, readable code
- Follow PEP 8 style guidelines
- Add comments where necessary
- Update documentation if needed

**5. Test Your Changes**
```bash
# Run the training script to ensure it works
python -m src.train

# Test with different configurations
# Make sure the model loads and evaluates correctly
```

**6. Commit Your Changes**
```bash
git add .
git commit -m "Add: description of your changes"
```

>Commit Message Guidelines:
>- `Add:` for new features
>- `Fix:` for bug fixes
>- `Update:` for updates to existing features
>- `Docs:` for documentation changes
>- `Refactor:` for code refactoring
>- `Test:` for adding tests

**7. Push to Your Fork**
```bash
git push origin feature/your-feature-name
```

**8. Create a Pull Request**
- Go to the original repository
- Click "New Pull Request"
- Select your fork and branch
- Provide a clear description of your changes
- Link any relevant issues

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>