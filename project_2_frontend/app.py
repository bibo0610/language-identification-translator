import json

import requests
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# CONFIG
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Language Identification & Translator",
    page_icon="🌐",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 20px;
    }

    .status-online {
        background-color: #e7f7ec;
        color: #167c3a;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }

    .status-offline {
        background-color: #fdecec;
        color: #b42318;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }

    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LANGUAGE SPEECH CODES
# =========================================================

speech_language_codes = {
    "ar": "ar-SA",
    "bg": "bg-BG",
    "de": "de-DE",
    "el": "el-GR",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "it": "it-IT",
    "ja": "ja-JP",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "sw": "sw-KE",
    "th": "th-TH",
    "tr": "tr-TR",
    "ur": "ur-PK",
    "vi": "vi-VN",
    "zh": "zh-CN"
}


# =========================================================
# CHECK BACKEND
# =========================================================

def check_backend():

    try:

        response = requests.get(
            f"{BACKEND_URL}/",
            timeout=3
        )

        return response.status_code == 200

    except Exception:

        return False


# =========================================================
# BROWSER TEXT-TO-SPEECH
# =========================================================

def browser_speech_button(
    text,
    language_code
):

    speech_code = speech_language_codes.get(
        language_code,
        language_code
    )

    # Chuyển text Python thành JavaScript an toàn
    text_js = json.dumps(text).replace(
        "</",
        "<\\/"
    )

    language_js = json.dumps(
        speech_code
    )

    html_code = f"""
    <button
        onclick="speakText()"
        style="
            width: 100%;
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 15px;
            cursor: pointer;
        "
    >
        🔊 Listen to Original
    </button>

    <script>

        function speakText() {{

            const text = {text_js};
            const language = {language_js};

            window.speechSynthesis.cancel();

            const speech =
                new SpeechSynthesisUtterance(text);

            speech.lang = language;

            window.speechSynthesis.speak(
                speech
            );
        }}

    </script>
    """

    components.html(
        html_code,
        height=60
    )


# =========================================================
# INITIAL SESSION STATE
# =========================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "text_area_input" not in st.session_state:
    st.session_state["text_area_input"] = ""


# =========================================================
# CLEAR FUNCTION
# =========================================================

def clear_all():

    # Xóa kết quả
    st.session_state["result"] = None

    # Xóa nội dung ô nhập
    st.session_state["text_area_input"] = ""


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🌐 Language Identification & Translator'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Automatically detect the language of a text '
    'and translate it into Vietnamese.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# BACKEND STATUS
# =========================================================

backend_online = check_backend()

if backend_online:

    st.markdown(
        '<div class="status-online">'
        '● Backend Online'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="status-offline">'
        '● Backend Offline'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# INPUT
# =========================================================

text_input = st.text_area(
    "Enter text",
    height=180,
    placeholder="Example: Hello, how are you today?",
    key="text_area_input"
)


# =========================================================
# BUTTONS
# =========================================================

button_col1, button_col2 = st.columns(
    [4, 1]
)


with button_col1:

    detect_button = st.button(
        "🔍 Detect & Translate",
        type="primary",
        use_container_width=True
    )


with button_col2:

    st.button(
        "🧹 Clear",
        use_container_width=True,
        on_click=clear_all
    )


# =========================================================
# PROCESS
# =========================================================

if detect_button:

    if not backend_online:

        st.error(
            "Backend is offline. "
            "Please start Project 1 first."
        )

    elif not text_input.strip():

        st.warning(
            "Please enter some text."
        )

    else:

        try:

            with st.spinner(
                "Detecting language and translating..."
            ):

                response = requests.post(
                    f"{BACKEND_URL}/process-text",
                    json={
                        "text": text_input
                    },
                    timeout=120
                )

            if response.status_code == 200:

                result = response.json()

                st.session_state["result"] = result

                st.success(
                    "Language detected and "
                    "translation completed successfully."
                )

            else:

                st.error(
                    "Backend error: "
                    + response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to backend."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The request took too long."
            )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )


# =========================================================
# RESULT
# =========================================================

if st.session_state["result"]:

    result = st.session_state["result"]

    st.divider()

    st.markdown(
        '<div class="result-title">'
        'Result'
        '</div>',
        unsafe_allow_html=True
    )

    column1, column2 = st.columns(
        2,
        gap="large"
    )


    # =====================================================
    # ORIGINAL
    # =====================================================

    with column1:

        st.markdown(
            "### 📝 Original"
        )

        st.info(
            f"Detected Language: "
            f"**{result['language']}** "
            f"({result['language_code']})"
        )

        st.text_area(
            "Original Text",
            value=result["original_text"],
            height=180,
            disabled=True,
            key="original_output"
        )

        browser_speech_button(
            result["original_text"],
            result["language_code"]
        )


    # =====================================================
    # VIETNAMESE
    # =====================================================

    with column2:

        st.markdown(
            "### 🇻🇳 Vietnamese"
        )

        st.info(
            "Target Language: "
            "**Vietnamese (vi)**"
        )

        st.text_area(
            "Vietnamese Translation",
            value=result["translated_text"],
            height=180,
            disabled=True,
            key="translated_output"
        )

        vietnamese_audio_button = st.button(
            "🔊 Listen to Vietnamese",
            use_container_width=True
        )


        # =================================================
        # VIETNAMESE TEXT TO SPEECH
        # =================================================

        if vietnamese_audio_button:

            try:

                with st.spinner(
                    "Generating Vietnamese audio..."
                ):

                    audio_response = requests.post(
                        f"{BACKEND_URL}/tts",
                        json={
                            "text":
                                result["translated_text"]
                        },
                        timeout=120
                    )

                if audio_response.status_code == 200:

                    st.audio(
                        audio_response.content,
                        format="audio/wav"
                    )

                else:

                    st.error(
                        "TTS error: "
                        + audio_response.text
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to backend."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The TTS request took too long."
                )

            except Exception as e:

                st.error(
                    f"TTS error: {e}"
                )