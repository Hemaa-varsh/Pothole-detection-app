# 🕳️ Real-Time Pothole Detection & Smart Road Monitoring System

AI-powered web application that detects potholes in road images using a custom-trained YOLOv8 model, classifies severity, and generates a Road Health Score — built as part of the Edunet Foundation × IBM SkillsBuild × AICTE AI Internship 2026.

🔗 **Live App:** [pothole-hema.streamlit.app](https://pothole-hema.streamlit.app)

---

## 📸 Preview

![App Screenshot](potholescreenshot.png)

---

## 🎯 Features

- 🔍 Real-time pothole detection on uploaded road images
- 🎨 Severity classification — Small / Medium / Large (color-coded bounding boxes)
- 🛣️ Road Health Score (0–100) with letter grade (A–F)
- 🎛️ Adjustable confidence threshold
- ⬇️ Downloadable annotated results
- 🌙 Custom dark dashboard UI

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Precision | 93.9% |
| Recall | 82.8% |
| F1 Score | 88.0% |
| mAP50 | 84.0% |
| mAP50-95 | 48.6% |
| Inference Speed | ~16ms (63 FPS) |

Model: **YOLOv8s**, fine-tuned via transfer learning on a custom 1,455-image pothole dataset (Roboflow).

---

## 🛠️ Tech Stack

`Python` · `YOLOv8 (Ultralytics)` · `OpenCV` · `Streamlit` · `NumPy` · `Pillow` · `Google Colab`

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/pothole-detection-app.git
cd pothole-detection-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

pothole-detection-app/

├── app.py              # Streamlit application
├── best.pt              # Trained YOLOv8 model weights
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── config.toml      # Dark theme configuration
└── README.md

---

## 👩‍💻 Author

**Hema**
Electronics and Communication Engineering (ECE), First Year
RMD Engineering College
Edunet Foundation × IBM SkillsBuild × AICTE AI Internship 2026
