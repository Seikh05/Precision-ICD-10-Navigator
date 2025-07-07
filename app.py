import gradio as gr
import pandas as pd
import numpy as np
import yaml
import re
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import requests
from io import BytesIO
import torch

# --- Configuration (Adjust paths for Hugging Face Space) ---
CONFIG_FILE = "config.yaml"
ICD10_CODES_FILE = "https://huggingface.co/datasets/Neural-Nook/icd10-codes-data/resolve/main/ICD10codes.csv"
ICD_EMBEDDINGS_FILE = "https://huggingface.co/datasets/Neural-Nook/icd10-codes-data/resolve/main/icd10_embeddings.npy"

# Load config globally
try:
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found. Please ensure it exists.")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing configuration file '{CONFIG_FILE}': {e}")


# --- Class Definitions (Keep these as they are from previous successful version) ---
class Preprocessor:
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_icd10_dataframe(self, df, code_col, desc_col):
        df_cleaned = df.dropna(subset=[code_col, desc_col]).copy()
        df_cleaned[desc_col] = df_cleaned[desc_col].apply(self.preprocess_text)
        df_cleaned = df_cleaned[df_cleaned[desc_col] != ''].drop_duplicates(subset=[code_col, desc_col])
        return df_cleaned

class SimilarityMatcher:
    def __init__(self, config_dict, icd_df_processed):
        self.config = config_dict
        self.model_name = self.config['SENTENCE_TRANSFORMER_MODEL_NAME']
        self.icd_data_df = icd_df_processed
        self.icd_descriptions_list = self.icd_data_df[self.config['ICD_DESCRIPTION_COL']].tolist()
        self.model = self._load_model()
        self.icd_embeddings = self._load_icd_embeddings_from_url()

    def _load_model(self):
        print(f"Loading Sentence Transformer model: {self.model_name}...")
        try:
            return SentenceTransformer(self.model_name)
        except Exception as e:
            raise Exception(f"Failed to load Sentence Transformer model '{self.model_name}'. Error: {e}")

    def _load_icd_embeddings_from_url(self):
        print(f"Attempting to download and load embeddings from URL: {ICD_EMBEDDINGS_FILE}")
        try:
            response = requests.get(ICD_EMBEDDINGS_FILE, stream=True)
            response.raise_for_status()
            embeddings_data = BytesIO(response.content)
            embeddings = np.load(embeddings_data, allow_pickle=False) 
            print(f"Successfully loaded embeddings of shape: {embeddings.shape} from URL.")
            if embeddings.shape[0] != len(self.icd_descriptions_list):
                print(f"WARNING: Mismatch between number of embeddings ({embeddings.shape[0]}) and ICD descriptions ({len(self.icd_descriptions_list)}). This might lead to incorrect matches.")
            return embeddings
        except requests.exceptions.RequestException as e:
            raise Exception(
                f"ERROR: Could not download embeddings from URL: {ICD_EMBEDDINGS_FILE}. "
                f"Please check the URL and its public accessibility. Error details: {e}"
            )
        except Exception as e:
            raise Exception(
                f"ERROR: Failed to load embeddings from downloaded content from {ICD_EMBEDDINGS_FILE}. "
                f"Ensure it's a valid .npy file. Error details: {e}"
            )

    def get_semantic_matches(self, input_text, num_results=5):
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Sentence Transformer model not loaded in SimilarityMatcher.")
        if not hasattr(self, 'icd_embeddings') or self.icd_embeddings is None:
            raise RuntimeError("ICD embeddings not loaded in SimilarityMatcher.")

        input_embedding = self.model.encode(input_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(input_embedding, self.icd_embeddings)[0]
        
        top_k = min(num_results, len(self.icd_data_df))
        top_n_indices = torch.topk(cosine_scores, k=top_k).indices.tolist()

        results = []
        for i in top_n_indices:
            try:
                code = self.icd_data_df.iloc[i][self.config['ICD_CODE_COL']]
                description = self.icd_data_df.iloc[i][self.config['ICD_DESCRIPTION_COL']]
                score = cosine_scores[i].item()
                results.append({
                    'code': code,
                    'description': description,
                    'score': score
                })
            except IndexError:
                print(f"Warning: Index {i} out of bounds for icd_data_df. Skipping.")
                continue
            except KeyError as ke:
                print(f"Warning: Missing expected column in icd_data_df: {ke}. Skipping.")
                continue
        return results

class ICD10Mapper:
    def __init__(self, config_dict, icd_df_raw_for_mapper):
        self.config = config_dict
        self.preprocessor = Preprocessor()
        print("\nLoading and preparing ICD-10 data for the mapper (one-time setup)...")
        self.icd_df_original = icd_df_raw_for_mapper.copy()
        self.icd_code_col = self.config['ICD_CODE_COL']
        self.icd_description_col = self.config['ICD_DESCRIPTION_COL']
        self.icd_df_processed = self.preprocessor.clean_icd10_dataframe(
            icd_df_raw_for_mapper, self.icd_code_col, self.icd_description_col
        )
        print(f"Processed {len(self.icd_df_processed)} unique ICD-10 entries.")
        self.similarity_matcher = SimilarityMatcher(config_dict, self.icd_df_processed)
        print("ICD-10 data preparation complete for mapper.")
        self.semantic_similarity_threshold = self.config['SEMANTIC_SIMILARITY_THRESHOLD']
        self.fuzzy_match_threshold = self.config['FUZZY_MATCH_THRESHOLD']
        self.num_alternatives = self.config['NUM_ALTERNATIVES']

    def _get_initial_match(self, diagnosis):
        processed_diagnosis = self.preprocessor.preprocess_text(diagnosis)
        if not processed_diagnosis: return None
        direct_match = self.icd_df_processed[self.icd_df_processed[self.icd_description_col] == processed_diagnosis]
        if not direct_match.empty:
            return {
                'code': direct_match.iloc[0][self.icd_code_col], 
                'description': direct_match.iloc[0][self.icd_description_col],
                'justification': "Direct string match", 
                'score': 1.0, 
                'alternatives': []
            }
        best_fuzzy_score = 0
        best_fuzzy_match = None
        for idx, row in self.icd_df_processed.iterrows():
            score = fuzz.token_set_ratio(processed_diagnosis, row[self.icd_description_col])
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_match = {
                    'code': row[self.icd_code_col], 
                    'description': row[self.icd_description_col],
                    'justification': f"Fuzzy match (score: {best_fuzzy_score:.2f})",
                    'score': best_fuzzy_score / 100.0, 
                    'alternatives': []
                }
        if best_fuzzy_score >= self.fuzzy_match_threshold and best_fuzzy_match:
            return best_fuzzy_match
        return None

    def map_diagnosis(self, diagnosis):
        initial_match = self._get_initial_match(diagnosis)
        if initial_match:
            return initial_match
        processed_diagnosis = self.preprocessor.preprocess_text(diagnosis)
        if not processed_diagnosis:
             return {
                 'code': 'N/A', 'description': 'No suitable match found',
                 'justification': 'Empty diagnosis after preprocessing.', 'score': 0.0, 'alternatives': []
             }
        semantic_matches = self.similarity_matcher.get_semantic_matches(
            processed_diagnosis, num_results=self.num_alternatives + 1
        )
        if not semantic_matches:
            return {
                'code': 'N/A', 'description': 'No suitable match found',
                'justification': 'No direct, fuzzy, or semantic match found above thresholds.',
                'score': 0.0, 'alternatives': []
            }
        best_semantic_match = semantic_matches[0]
        alternatives = semantic_matches[1:]
        justification = f"Semantic similarity match (score: {best_semantic_match['score']:.2f})"
        if best_semantic_match['score'] < self.semantic_similarity_threshold:
            justification += ". Confidence below threshold, review carefully."
        return {
            'code': best_semantic_match['code'], 
            'description': best_semantic_match['description'],
            'justification': justification, 
            'score': best_semantic_match['score'],
            'alternatives': [
                {'code': alt['code'], 'description': alt['description'], 'score': alt['score']}
                for alt in alternatives
            ]
        }

# --- Global Initialization (runs once when the Space starts) ---
print("--- Initializing ICD-10 Mapper for Gradio App ---")
try:
    icd_df_raw = pd.read_csv(ICD10_CODES_FILE)
    print(f"Successfully loaded raw ICD-10 data from {ICD10_CODES_FILE}")
except Exception as e:
    print(f"Error: Could not load raw ICD-10 data from {ICD10_CODES_FILE}. Please check URL or file existence. Details: {e}")
    raise

mapper = ICD10Mapper(config, icd_df_raw)
print("--- ICD-10 Mapper Initialization Complete ---")

# --- Gradio Interface Function ---
def map_diagnosis_for_gradio(diagnosis_text):
    if not diagnosis_text or diagnosis_text.strip() == "":
        return "Please enter a diagnosis to map."
    result = mapper.map_diagnosis(diagnosis_text)
    output_markdown = f"### Mapped Result:\n" \
                      f"- **Mapped ICD-10 Code:** `{result['code']}`\n" \
                      f"- **Description:** `{result['description']}`\n" \
                      f"- **Similarity Score:** `{result['score']:.2f}`\n" \
                      f"- **Justification:** `{result['justification']}`\n\n"
    if result['alternatives']:
        output_markdown += "### Alternatives:\n"
        for i, alt in enumerate(result['alternatives']):
            output_markdown += f"- `{alt['code']}`: {alt['description']} (Score: `{alt['score']:.2f}`)\n"
    else:
        output_markdown += "### Alternatives:\n- No alternatives found."
    return output_markdown


# --- Custom Theme for a Cyanic Blue Look ---
custom_cyan_theme = gr.themes.Soft(
    primary_hue="cyan",    # Primary color for buttons, etc.
    secondary_hue="blue",  # Secondary color for accents
    neutral_hue="gray",   # Grey for neutral elements
    font=(gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"),
    text_size=gr.themes.sizes.text_md,
).set(
    button_primary_background_fill="*primary_500", # Main buttons will use the primary 500 shade
    button_primary_text_color="white",
    # Optionally, set secondary button colors explicitly if needed
    button_secondary_background_fill="*secondary_500",
    button_secondary_text_color="white",
    border_color_accent="*primary_600",
)

# --- Custom CSS for overall layout ---
custom_app_css = """
/* NEW: Add gap between columns in a row */
.gradio-container > div > .gr-row { /* Targets rows directly inside the main container */
    gap: 100px; /* Adjust this value (e.g., 20px, 40px) for more or less space */
    align-items: flex-start; /* Optional: Aligns items to the top if they have different heights */
}
"""
# --- Gradio Blocks Interface Setup for Custom Layout and Footer ---
with gr.Blocks(theme=custom_cyan_theme, title="Precision ICD-10 Navigator", css=custom_app_css) as demo:
    # 1. Unique and Good Title / Description
    gr.Markdown(
        """
        # 🏥 Precision ICD-10 Navigator: Symptom-to-Code Mapper
        *Effortlessly convert free-text patient diagnoses and symptoms into precise ICD-10 codes.*

        """
    )


       
    with gr.Row():
    # Input Section (left)
        with gr.Column(scale=1):  # 1 part width
            diagnosis_input = gr.Textbox(
                lines=6,
                label="📝 Patient Diagnosis or Symptoms",
                placeholder="e.g., 'acute appendicitis with generalized peritonitis'",
                interactive=True,
                elem_classes=["input-box"]
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Find ICD-10 Code", variant="primary")
                clear_btn = gr.Button("🧹 Clear", variant="secondary")

        # Output Section (right)
        with gr.Column(scale=0.9):  
            output_display = gr.Markdown(
                label="📋 Mapping Results & Alternatives",
                elem_classes=["output-box"]
            )


    # Examples below the main section
    gr.Examples(
        examples=[
            ["headache"],
            ["common cold"],
            ["fracture of the distal radius"],
            ["type 2 diabetes mellitus with complications"],
            ["viral infection"],
            ["chest pain"]
        ],
        inputs=diagnosis_input,
        outputs=output_display,
        fn=map_diagnosis_for_gradio,
        cache_examples=True # Recommended for better performance with examples
    )

    # 2. Add a footer with "Made by (Your Name)" and a link
    gr.Markdown(
        f"""
        ---
        <p style="text-align: center; color: gray; font-size: 0.9em;">
            Made with ❤️ by <strong>Mustakim</strong> <br>
            <a href="https://huggingface.co/spaces/Neural-Nook/icd10-diagnosis-mapper" target="_blank" style="color: {custom_cyan_theme.primary_500}; text-decoration: none;">Visit this Space on Hugging Face</a>
            <br><i>Data last updated: July 2025</i>
        </p>
        """
    )

    # Link button actions (for Blocks, you link actions to components using .click())
    submit_btn.click(
        fn=map_diagnosis_for_gradio,
        inputs=diagnosis_input,
        outputs=output_display
    )
    
    # Clear button action: clears both input and output
    clear_btn.click(
        fn=lambda: ("", ""), # Returns a tuple of empty strings for the outputs
        inputs=[],
        outputs=[diagnosis_input, output_display]
    )

# Launch the Gradio app (using 'demo' instead of 'iface')
if __name__ == "__main__":
    demo.launch(share=False)