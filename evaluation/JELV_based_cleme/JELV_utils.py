import os
import torch
import torch.nn as nn
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
import warnings
import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

# ---------- Configuration Constants ----------
MODEL_NAME       = "microsoft/deberta-v3-large"
SAVED_MODEL_PATH = "/data/zyh/JELV/evaluation/JudgeModel/best_model"
MAX_LENGTH       = 256
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- DeBERTa Classifier Definition ----------
class FeatureEnhancedDebertaClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_features=2):
        super().__init__()
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, return_dict=True
        )
        # Optional dropout tuning
        self.deberta.config.update({
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1
        })
        hidden_size = self.deberta.config.hidden_size
        if num_features > 0:
            self.feature_projection = nn.Linear(num_features, 64)
            self.classifier         = nn.Linear(hidden_size + 64, 2)
            # bypass original classifier
            self.deberta.classifier = nn.Identity()
        self.num_features = num_features

    def forward(self, input_ids=None, attention_mask=None, features=None, **kwargs):
        output = self.deberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = output.logits
        if features is not None and self.num_features > 0:
            if isinstance(features, list):
                features = torch.tensor(features, dtype=torch.float, device=input_ids.device)
            proj   = self.feature_projection(features)
            logits = self.classifier(torch.cat([logits, proj], dim=1))
        return {"logits": logits}

# ---------- Inference Components Initialization ----------
def init_inference_components():
    """
    Load and return (tokenizer, model, SBERT, GPT2 model, GPT2 tokenizer).
    """
    # Main classifier
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model     = FeatureEnhancedDebertaClassifier().to(DEVICE).eval()
    state_dict = load_file(os.path.join(SAVED_MODEL_PATH, "model.safetensors"), device="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Semantic similarity model
    sbert = SentenceTransformer(
        "snunlp/KR-SBERT-V40K-klueNLI-augSTS", local_files_only=True
    ).to(DEVICE)

    # Language model for cross-entropy features
    gpt_model     = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).to(DEVICE)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2", clean_up_tokenization_spaces=False
    )

    return tokenizer, model, sbert, gpt_model, gpt_tokenizer

# ---------- Cross-Entropy Feature Calculation ----------
def calculate_cross_entropy(text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device=DEVICE) -> float:
    """
    Compute average cross-entropy loss of a single text sequence.
    """
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(tokens, labels=tokens).loss
    return loss.item()
