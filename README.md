# 🏥 ICD-10 Diagnosis Mapper

A smart Gradio-based AI app that maps free-text clinical diagnoses to standardized **ICD-10 codes**, using semantic embeddings and fuzzy logic.

---
## 🎨 App UI Preview

![ICD-10 Mapper UI](https://drive.google.com/uc?id=1mHV1grcGoKl7SYwBeFPahfyX56Ar0meW)

## 🚀 Features

- 🔍 Input free-form clinical descriptions
- 💡 Smart semantic matching with **Sentence Transformers**
- 🧾 Fuzzy logic for approximate textual matches
- 🔗 Hugging Face-hosted **ICD-10 data & embeddings**
- 📊 Full pipeline reproducible in **Google Colab**
- 🎨 Beautiful Gradio UI with live predictions and examples

---

## 📂 Project Structure

### 🖥️ Local Project Layout

```bash
icd10_diagnosis_mapping/
├── data/
│   ├── raw/
│   │   ├── ICD10codes.csv
│   │   └── Diagnoses_list.xlsx
│   ├── processed/
│   └── output/
├── models/
│   └── embeddings/
├── notebooks/
│   └── ICD10_Mapping_Colab.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── icd10_mapper.py
│   ├── similarity_matcher.py
│   └── main.py
├── config/
│   └── config.yaml
├── run_pipeline.py
├── requirements.txt
└── README.md
````

---

### 📁 Colab-Compatible Google Drive Structure

If using [Google Colab](https://colab.research.google.com):

```
My Drive/
└── ICD10_Mapping_Project/
    ├── data/
    │   ├── raw/
    │   │   ├── ICD10codes.csv
    │   │   └── Diagnoses_list.xlsx
    │   ├── processed/
    │   └── output/
    ├── models/
    │   └── embeddings/
    ├── notebooks/
    │   └── ICD10_Mapping_Colab.ipynb
    └── config/
        └── config.yaml
```

---

## 🧪 Try the Full Pipeline on Google Colab

🎯 Click the badge below to open the full working pipeline notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19BeTdJyKUaMa2MwK5kUrwMHq6ErXLCbr?usp=sharing)

The Colab notebook includes:

* 📥 Loading ICD-10 and patient diagnosis data
* 🧠 Sentence Transformer embedding generation
* 🔍 Semantic similarity + fuzzy matching
* 📤 Exporting mapped results as `.csv` and `.json`

---

## 🧠 Training & Pipeline Summary

1. **Load Raw Data**: `ICD10codes.csv` and `Diagnoses_list.xlsx` from `data/raw/`
2. **Generate Embeddings** using `all-MiniLM-L6-v2`

   * Saved to `models/embeddings/icd10_embeddings.npy`
3. **Perform Semantic Matching** with cosine similarity
4. **Output Results**

   * `mapped_diagnoses.csv`: main mapping table
   * `mapping_report.json`: confidence and ambiguity report

### Example CLI Output

```bash
(venv) (.venv) C:\Users\91955\OneDrive\Desktop\ICD10>python run_pipeline.py
Loading Sentence Transformer model: all-MiniLM-L6-v2
--- Starting ICD-10 Diagnosis Mapping Pipeline ---

Step 1: Loading raw data...
Loaded ICD-10 codes from: data/raw\ICD10codes.csv
Loaded patient diagnoses from: data/raw\Diagnoses_list.xlsx

Step 2: Validating raw data...
Preparing ICD-10 data...
Generating ICD-10 embeddings (this may take a while)...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2241/2241 [06:21<00:00,  5.88it/s] 
Saved ICD-10 embeddings to: models\embeddings\icd10_embeddings.npy
ICD-10 data preparation complete.

Step 4: Preprocessing patient diagnoses...

Step 5: Mapping diagnoses to ICD-10 codes...

Step 6: Saving results...
Saved mapped diagnoses to: data/output\mapped_diagnoses.csv

Pipeline completed successfully!
Saved mapping report to: data/output\mapping_report.json
```

---

## 🗃️ Data Files Used

### 📄 [`ICD10codes.csv`](https://huggingface.co/datasets/Neural-Nook/icd10-codes-data/blob/main/ICD10codes.csv)

* Official ICD-10 codes and descriptions.
### 🧠 [`Diagnoses_list.xlsx`](https://docs.google.com/spreadsheets/d/1O2wW-wQukh2F2o4_w7AmM2mS2wFwMGzYCdIGh54MT8Q/edit?gid=0#gid=0))

* Sentence Transformer embeddings for ICD-10 descriptions.

### 🧠 [`icd10_embeddings.npy`](https://huggingface.co/datasets/Neural-Nook/icd10-codes-data/resolve/main/icd10_embeddings.npy)

* It holds all the symptomes/diagnoses list to be mapped.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/icd10-diagnosis-mapper.git
cd icd10-diagnosis-mapper
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

# 🚀 Deploying to Hugging Face Spaces

You can host this project as a **public or private app** on [Hugging Face Spaces](https://huggingface.co/spaces), using **Gradio** as the interface.

---

### 🛠️ Deployment Steps

1. **Create a Hugging Face account**
   Sign up or log in at [https://huggingface.co](https://huggingface.co)

2. **Go to** [https://huggingface.co/spaces](https://huggingface.co/spaces)
   Click **“Create new Space”**

3. **Configure your space**

   * **SDK**: Select `Gradio`
   * **Visibility**: Choose `Public` or `Private`
   * **Space name**: Example: `mustakim/icd10-diagnosis-mapper`

4. **Clone your Space locally**:

   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/your-username/icd10-diagnosis-mapper
   cd icd10-diagnosis-mapper
   ```

5. **Add your files**
   Make sure the following are present in the repo:

   * `app.py` (Gradio entry point)
   * `requirements.txt`
   * `README.md`
   * `config.yaml` (optional, for model/code settings)
   * Precomputed files (if applicable): `icd10_embeddings.npy`

6. **Push to your Space**

   ```bash
   git add .
   git commit -m "Initial push"
   git push
   ```

7. **Your app will automatically deploy!**
   Hugging Face will build the environment, install dependencies, and launch your app using Gradio.

---

### ✅ Tips

* Add a `.gitattributes` file with:

  ```
  *.npy filter=lfs diff=lfs merge=lfs -text
  ```

  if you're using Git LFS to track large files.

* If your app takes time to load (e.g. loading `.npy` files or models), consider using `gr.load()` callbacks to defer heavy tasks until first use.

---

## 🌐 Live Demo

👉 **[Launch Hugging Face Demo](https://huggingface.co/spaces/Neural-Nook/icd10-diagnosis-mapper)**

---

## 🧪 Example Inputs

* `type 2 diabetes mellitus with complications`
* `viral infection`
* `chest pain`
* `abnormal weight loss`

---

## 👨‍💻 Author

**Made by Mustakim**

> Powered by open-source tools & Hugging Face ❤️

---

## 📄 License

This project is released under the **MIT License**.
