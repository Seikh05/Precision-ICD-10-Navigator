# 🏥 ICD-10 Diagnosis Mapper

A smart Gradio-based AI app that maps free-text clinical diagnoses to standardized **ICD-10 codes**, using semantic embeddings and fuzzy logic.

---

## 🚀 Features

- 🔍 Input free-form clinical descriptions
- 💡 Smart semantic matching with **Sentence Transformers**
- 🧾 Fuzzy logic for approximate textual matches
- 🔗 Hugging Face-hosted **ICD-10 data & embeddings**
- 🎨 Beautiful Gradio UI with live predictions and examples

---

## 🗃️ Data Files Used

### 📄 [`ICD10codes.csv`](https://huggingface.co/datasets/Neural-Nook/icd10-codes-data/blob/main/ICD10codes.csv)
- Contains raw ICD-10 codes and official medical descriptions.
- Used to compare user input against real-world diagnoses.
- Loaded directly via URL using `pandas.read_csv()`.

### 🧠 [`icd10_embeddings.npy`](https://huggingface.co/datasets/Neural-Nook/icd10-codes-data/resolve/main/icd10_embeddings.npy)
- Pre-computed sentence embeddings for each ICD-10 description.
- Created using a Sentence Transformer (`all-MiniLM-L6-v2`).
- Loaded with `numpy.load()` from Hugging Face.

---

## 📂 Project Structure

```bash
ICD10_DIAGNOSIS_MAPPER/
├── app.py              # Main Gradio app
├── config.yaml         # Settings for models and thresholds
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── .gitattributes      # Git LFS tracking (if needed)
````

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/icd10-diagnosis-mapper.git
cd icd10-diagnosis-mapper
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
python app.py
```

This will launch the Gradio UI at:

```
http://localhost:7860
```

---

## 🛰️ Deploy to Hugging Face Spaces

You can host your app publicly using [Hugging Face Spaces](https://huggingface.co/spaces):

### ✅ Steps:

1. **Create a Hugging Face account** (if not already)
2. **Go to:** [https://huggingface.co/spaces](https://huggingface.co/spaces)
3. Click **"Create New Space"**
4. Choose:

   * SDK: **Gradio**
   * Visibility: **Public or Private**
5. Clone the repo into your space:

   * Either **upload files manually**
   * Or connect your GitHub repo with:

     ```bash
     git remote add origin https://huggingface.co/spaces/your-username/icd10-diagnosis-mapper
     git push -u origin main
     ```
6. Make sure your **`app.py`** is the entry point
7. Add these required files:

   * `requirements.txt`
   * `config.yaml`

Done! Hugging Face will automatically launch your app.

---

## 🌐 Demo on Hugging Face

👉 **[Launch ICD-10 Mapper Demo](https://huggingface.co/spaces/Neural-Nook/icd10-diagnosis-mapper)**

---

## 🧠 Technologies Used

| Component                | Description                                |
| ------------------------ | ------------------------------------------ |
| 🔎 Sentence Transformers | `all-MiniLM-L6-v2` for semantic embeddings |
| 🧮 NumPy & Pandas        | Efficient data loading and handling        |
| 🔤 FuzzyWuzzy            | Token-based string similarity              |
| 🎨 Gradio                | Frontend interface for interaction         |
| ☁️ Hugging Face          | Dataset and App hosting                    |

---

## 🧪 Example Inputs

* `type 2 diabetes mellitus with complications`
* `viral infection`
* `fracture of the distal radius`
* `chest pain`
* `headache`

---

## 👨‍💻 Author

**Made by Mustakim**

> Powered by open-source tech & Hugging Face ❤️

---

## 📄 License

This project is released under the **MIT License**.

