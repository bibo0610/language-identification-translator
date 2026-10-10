import re
import html
import io
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import torch

from scipy.io.wavfile import write

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    VitsTokenizer,
    VitsModel,
    set_seed
)


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Language Identification API",
    description=(
        "API nhận diện ngôn ngữ văn bản, "
        "dịch sang tiếng Việt và đọc văn bản tiếng Việt"
    ),
    version="1.0"
)


# =========================================================
# LOAD LANGUAGE IDENTIFICATION MODEL
# =========================================================

print("Loading language identification model...")

vectorizer = joblib.load(
    MODEL_DIR / "final_tfidf_vectorizer.joblib"
)

model = joblib.load(
    MODEL_DIR / "final_language_model.joblib"
)

print("Language identification model loaded!")


# =========================================================
# LOAD TRANSLATION MODEL
# =========================================================

TRANSLATION_MODEL = (
    "facebook/nllb-200-distilled-600M"
)

print("Loading translation model...")

translation_tokenizer = (
    AutoTokenizer.from_pretrained(
        TRANSLATION_MODEL
    )
)

translation_model = (
    AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL
    )
)

translation_model.eval()

print("Translation model loaded!")


# =========================================================
# LOAD VIETNAMESE TEXT-TO-SPEECH MODEL
# =========================================================

TTS_MODEL = "facebook/mms-tts-vie"

print("Loading Vietnamese TTS model...")

tts_tokenizer = (
    VitsTokenizer.from_pretrained(
        TTS_MODEL
    )
)

tts_model = (
    VitsModel.from_pretrained(
        TTS_MODEL
    )
)

tts_model.eval()

print("Vietnamese TTS model loaded!")


# =========================================================
# LANGUAGE NAMES
# =========================================================

language_names = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese"
}


# =========================================================
# LANGUAGE CODE MAPPING FOR NLLB
# =========================================================

nllb_language_codes = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "sw": "swh_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "zh": "zho_Hans"
}


# =========================================================
# REQUEST MODELS
# =========================================================

class TextInput(BaseModel):
    text: str


class TranslationInput(BaseModel):
    text: str
    source_language: str


# =========================================================
# PREPROCESSING
# =========================================================

def clean_text(text):

    text = str(text)

    # Convert HTML entities
    # Example: &amp; -> &
    text = html.unescape(text)

    # Normalize Unicode
    text = unicodedata.normalize(
        "NFC",
        text
    )

    # Remove URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# LANGUAGE DETECTION FUNCTION
# =========================================================

def predict_language(text):

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        text_vector
    )[0]

    return prediction, cleaned_text


# =========================================================
# TRANSLATION FUNCTION
# =========================================================

def translate_to_vietnamese(
    text,
    source_language
):

    # Nếu văn bản đã là tiếng Việt
    if source_language == "vi":
        return text

    if (
        source_language
        not in nllb_language_codes
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported source language."
            )
        )

    try:

        source_code = (
            nllb_language_codes[
                source_language
            ]
        )

        # Set source language
        translation_tokenizer.src_lang = (
            source_code
        )

        # Tokenize input
        inputs = translation_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Token đại diện tiếng Việt
        vietnamese_token_id = (
            translation_tokenizer
            .convert_tokens_to_ids(
                "vie_Latn"
            )
        )

        # Generate translation
        with torch.no_grad():

            translated_tokens = (
                translation_model.generate(
                    **inputs,
                    forced_bos_token_id=(
                        vietnamese_token_id
                    ),
                    max_length=512
                )
            )

        # Decode output
        translated_text = (
            translation_tokenizer
            .batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )[0]
        )

        return translated_text

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Translation failed: "
                + str(e)
            )
        )


# =========================================================
# TEXT-TO-SPEECH FUNCTION
# =========================================================

def generate_vietnamese_speech(text):

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    try:

        inputs = tts_tokenizer(
            text=cleaned_text,
            return_tensors="pt"
        )

        # Giữ kết quả ổn định hơn giữa các lần chạy
        set_seed(42)

        with torch.no_grad():

            output = tts_model(
                **inputs
            )

        waveform = (
            output.waveform
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        audio_buffer = io.BytesIO()

        write(
            audio_buffer,
            tts_model.config.sampling_rate,
            waveform
        )

        audio_buffer.seek(0)

        return audio_buffer

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Text-to-Speech failed: "
                + str(e)
            )
        )


# =========================================================
# ENDPOINT 1 - HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": (
            "Language Identification API "
            "is running"
        )
    }


# =========================================================
# ENDPOINT 2 - DETECT LANGUAGE
# =========================================================

@app.post("/detect-language")
def detect_language(
    data: TextInput
):

    prediction, cleaned_text = (
        predict_language(
            data.text
        )
    )

    return {
        "original_text": data.text,
        "clean_text": cleaned_text,
        "language_code": prediction,
        "language": language_names.get(
            prediction,
            prediction
        )
    }


# =========================================================
# ENDPOINT 3 - TRANSLATE TO VIETNAMESE
# =========================================================

@app.post("/translate")
def translate_text(
    data: TranslationInput
):

    cleaned_text = clean_text(
        data.text
    )

    if not cleaned_text:

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    if (
        data.source_language
        not in language_names
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported source language."
            )
        )

    translated_text = (
        translate_to_vietnamese(
            cleaned_text,
            data.source_language
        )
    )

    return {
        "original_text": data.text,

        "source_language_code":
            data.source_language,

        "source_language":
            language_names[
                data.source_language
            ],

        "target_language_code":
            "vi",

        "target_language":
            "Vietnamese",

        "translated_text":
            translated_text
    }


# =========================================================
# ENDPOINT 4 - DETECT AND TRANSLATE
# =========================================================

@app.post("/process-text")
def process_text(
    data: TextInput
):

    # Detect language
    prediction, cleaned_text = (
        predict_language(
            data.text
        )
    )

    # Translate to Vietnamese
    translated_text = (
        translate_to_vietnamese(
            cleaned_text,
            prediction
        )
    )

    return {
        "original_text":
            data.text,

        "clean_text":
            cleaned_text,

        "language_code":
            prediction,

        "language":
            language_names.get(
                prediction,
                prediction
            ),

        "target_language_code":
            "vi",

        "target_language":
            "Vietnamese",

        "translated_text":
            translated_text
    }


# =========================================================
# ENDPOINT 5 - VIETNAMESE TEXT TO SPEECH
# =========================================================

@app.post("/tts")
def text_to_speech(
    data: TextInput
):

    audio_buffer = (
        generate_vietnamese_speech(
            data.text
        )
    )

    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={
            "Content-Disposition":
                'attachment; filename="speech.wav"'
        }
    )