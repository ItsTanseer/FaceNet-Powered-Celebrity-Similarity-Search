# FaceTwin AI 🎭

**Find the bollywood celebrity you look most like using FaceNet embeddings and deep metric learning.**

##  Overview

FaceTwin AI is a celebrity look-alike detection system that uses **FaceNet**, **MTCNN**, and **cosine similarity** to identify celebrities whose facial features most closely resemble a user's uploaded image.

Instead of treating the task as a traditional classification problem, FaceTwin AI leverages **face embeddings** and **similarity search** to compare facial features in a high-dimensional embedding space.

---

##  Features

* Upload any image containing a face
* Automatic face detection using MTCNN
* Face embedding generation using FaceNet (InceptionResnetV1)
* Similarity-based celebrity matching
* Returns the Top-3 most similar celebrities
* Streamlit-based interactive web application
* Fast inference using precomputed celebrity embeddings

---

##  How It Works

```text
User Image
    ↓
MTCNN Face Detection
    ↓
Face Cropping & Alignment
    ↓
FaceNet Embedding Generation
    ↓
512-D Face Embedding
    ↓
Cosine Similarity Search
    ↓
Top Celebrity Matches
```

### Step 1: Face Detection

The uploaded image is passed through **MTCNN** to detect and crop the face.

### Step 2: Feature Extraction

The cropped face is passed through a pretrained **FaceNet (InceptionResnetV1)** model trained on the VGGFace2 dataset.

Output:

```text
Face Image
    ↓
FaceNet
    ↓
512-Dimensional Embedding Vector
```

### Step 3: Celebrity Database

For each celebrity:

* Multiple images are collected
* Embeddings are generated for each image
* A representative embedding is obtained by averaging all embeddings

```text
Celebrity Images
      ↓
FaceNet
      ↓
Embeddings
      ↓
Mean Embedding
```

### Step 4: Similarity Search

The user's embedding is compared against all celebrity embeddings using cosine similarity.

```text
Higher Similarity
        ↓
More Similar Facial Features
```





---

##  Tech Stack

### Machine Learning

* PyTorch
* FaceNet (InceptionResnetV1)
* MTCNN
* Scikit-learn

### Data Processing

* NumPy
* Pandas
* Pillow

### Frontend

* Streamlit

---

##  Dataset

The dataset consists of celebrity face images organized in the following structure:

```text
dataset/
│
├── Hrithik_Roshan/
│   ├── Hrithik_Roshan1.jpg
│   ├── Hrithik_Roshan2.jpg
│   └── ...
│
├── Shahid_Kapoor/
│   ├── Shahid_Kapoor1.jpg
│   ├── Shahid_Kapoor2.jpg
│   └── ...
│
└── ...
```

Each celebrity folder contains multiple facial images used to generate robust facial embeddings.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/FaceTwinAI.git

cd FaceTwinAI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

* FAISS-based similarity search
* Support for multiple faces in a single image
* Real-time webcam celebrity matching
* Gender and age-aware matching
* Celebrity profile display
* Confidence visualization charts
* Cloud deployment

---

## 🎯 Applications

* Entertainment and social media
* Face similarity search
* Face recognition systems
* Deep metric learning demonstrations
* Computer vision portfolio projects

---

## 📚 Concepts Used

* Deep Metric Learning
* Face Recognition
* Face Embeddings
* Cosine Similarity
* Transfer Learning
* Computer Vision
* Representation Learning

---

## 👨‍💻 Author

Developed using:

* PyTorch
* FaceNet
* MTCNN
* Streamlit

Built as a deep learning and computer vision project demonstrating face embedding generation and similarity-based celebrity matching.
