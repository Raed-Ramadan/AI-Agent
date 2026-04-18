import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from openai import OpenAI

# RAG imports
from rag_ingest import DocumentIngester
from rag_chunking import TextChunker
from rag_store import VectorStore
from rag_retriever import DocumentRetriever
from rag_prompting import RAGPromptBuilder


# =========================================================
# App Identity
# =========================================================

APP_TITLE = "ISO19650 Agent"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_KEYS_URL = "https://openrouter.ai/keys"

BASE_DIR = Path(__file__).resolve().parent
SUBJECTS_DIR = BASE_DIR / "subjects"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
ISO_KNOWLEDGE_DIR = KNOWLEDGE_DIR / "iso19650"

MAX_PAGES = 100


# =========================================================
# States
# =========================================================

STATES = [
    "awaiting_ui_language",
    "awaiting_teaching_language",
    "awaiting_guidance_focus",
    "awaiting_depth",
    "awaiting_material",
    "awaiting_pages",
    "teaching",
    "reviewing",
    "summarizing",
]


# =========================================================
# Options
# =========================================================

UI_LANGUAGE_OPTIONS = {
    "ar": "العربية",
    "en": "English",
}

TEACHING_LANGUAGE_OPTIONS = {
    "ar": "العربية",
    "en": "English",
}

GUIDANCE_FOCUS_OPTIONS = {
    "ar": {
        "iso_work_guidance": "إرشاد العمل وفق ISO 19650",
        "bim_workflow_guidance": "شرح BIM Workflow أو موضوع BIM",
        "problem_explanation": "شرح مشكلة أو نقطة غير واضحة",
        "general_learning": "شرح موضوع أو محتوى تعليمي عام",
    },
    "en": {
        "iso_work_guidance": "ISO 19650 Work Guidance",
        "bim_workflow_guidance": "BIM Workflow / BIM Topic Guidance",
        "problem_explanation": "Problem Explanation",
        "general_learning": "General Learning Topic",
    },
}

LEARNING_DEPTH_OPTIONS = {
    "ar": {
        "quick": "نظرة سريعة",
        "guided": "تعلم موجه",
        "deep": "تعلم عميق",
    },
    "en": {
        "quick": "Quick Overview",
        "guided": "Guided Learning",
        "deep": "Deep Learning",
    },
}


# =========================================================
# UI Texts
# =========================================================

UI_TEXTS = {
    "ar": {
        "app_title": "ISO19650 Agent",
        "subject_library": "مكتبة المواضيع",
        "start_new": "ابدأ موضوع جديد",
        "resume": "استئناف",
        "settings": "الإعدادات",
        "ui_language": "لغة واجهة المستخدم",
        "teaching_language": "لغة الشرح",
        "choose_ui_language": "اختر لغة واجهة المستخدم",
        "choose_teaching_language": "اختر لغة الشرح",
        "select_language": "اختر اللغة:",
        "confirm": "تأكيد",
        "choose_focus": "اختر نوع المساعدة المطلوبة",
        "select_focus": "نوع المساعدة:",
        "choose_depth": "اختر مستوى عمق الشرح",
        "select_depth": "مستوى الشرح:",
        "add_material": "إضافة المحتوى",
        "send_material": "ارسل المحتوى.",
        "input_method": "طريقة الإدخال:",
        "pasted_text": "نص ملصق",
        "upload_file": "رفع ملف",
        "paste_here": "الصق المحتوى هنا:",
        "upload_prompt": "ارفع ملف (.txt, .md)",
        "submit_material": "إرسال المحتوى",
        "file_error": "تعذر قراءة الملف. استخدم ملفًا نصيًا واضحًا مثل txt أو md.",
        "material_empty": "أضف محتوى أولًا قبل المتابعة.",
        "pages_question": "عايز أقسم الشرح على كام صفحة؟",
        "pages_label": "عدد الصفحات:",
        "max_pages_error": "Max value is 100",
        "processing_input": "جارٍ معالجة الطلب.",
        "api_key": "مفتاح OpenRouter API",
        "get_key": "احصل على المفتاح",
        "select_model": "اختر النموذج",
        "no_models": "لم يتم العثور على نماذج مجانية من OpenRouter. راجع المفتاح أو جرّب حسابًا آخر.",
        "temperature": "درجة الحرارة",
        "temperature_help": "القيمة الأقل = أكثر ثباتًا. القيمة الأعلى = أكثر مرونة.",
        "api_warning": "من فضلك أدخل API key صحيح واختر نموذجًا مجانيًا.",
        "your_message": "اكتب رسالتك",
        "next_button": "التالي",
        "generate_first_page": "جارٍ توليد الصفحة الأولى تلقائيًا...",
        "subject": "الموضوع",
        "language": "لغة الشرح",
        "depth": "العمق",
        "focus": "نوع المساعدة",
        "page": "الصفحة",
        "progress": "التقدم",
        "source_type": "نوع المصدر",
        "created_at": "تاريخ الإنشاء",
        "last_opened": "آخر فتح",
        "no_subject_loaded": "لا يوجد موضوع محمّل حاليًا.",
        "no_subjects": "لا توجد مواضيع محفوظة حتى الآن.",
        "status_new": "جديد",
        "status_in_progress": "قيد التقدم",
        "status_completed": "مكتمل",
        "download_txt": "تحميل TXT",
        "download_json": "تحميل JSON",
        "waiting_content": "في انتظار ظهور المحتوى التعليمي.",
        "use_commands": "يمكنك استخدام الشات بحرية أو الضغط على زر التالي.",
        "title_fallback": "موضوع بدون عنوان",
        "knowledge_loaded": "تم تحميل ملفات المعرفة الخاصة بـ ISO 19650.",
        "knowledge_missing": "لم يتم العثور على ملفات المعرفة الخاصة بـ ISO 19650.",
        "focus_hint": "هذا الاختيار يساعد الوكيل على فهم نوع الدعم المطلوب قبل بدء الشرح.",
    },
    "en": {
        "app_title": "ISO19650 Agent",
        "subject_library": "Subject Library",
        "start_new": "Start New Subject",
        "resume": "Resume",
        "settings": "Settings",
        "ui_language": "User Interface Language",
        "teaching_language": "Teaching Language",
        "choose_ui_language": "Choose User Interface Language",
        "choose_teaching_language": "Choose Teaching Language",
        "select_language": "Select language:",
        "confirm": "Confirm",
        "choose_focus": "Choose the kind of guidance you need",
        "select_focus": "Guidance type:",
        "choose_depth": "Choose learning depth",
        "select_depth": "Depth:",
        "add_material": "Add Material",
        "send_material": "Send the material.",
        "input_method": "Input method:",
        "pasted_text": "Pasted Text",
        "upload_file": "Upload File",
        "paste_here": "Paste your material here:",
        "upload_prompt": "Upload file (.txt, .md)",
        "submit_material": "Submit Material",
        "file_error": "Unable to read the file. Please use a clear text-based file like txt or md.",
        "material_empty": "Please add material before continuing.",
        "pages_question": "How many pages would you like me to divide the explanation into?",
        "pages_label": "Number of pages:",
        "max_pages_error": "Max value is 100",
        "processing_input": "Processing request.",
        "api_key": "OpenRouter API Key",
        "get_key": "Get API Key",
        "select_model": "Select Model",
        "no_models": "No free OpenRouter models were found. Check your key or try another account.",
        "temperature": "Temperature",
        "temperature_help": "Lower = more stable. Higher = more flexible.",
        "api_warning": "Please enter a valid API key and choose a free model.",
        "your_message": "Type your message",
        "next_button": "Next",
        "generate_first_page": "Generating the first page automatically...",
        "subject": "Subject",
        "language": "Teaching Language",
        "depth": "Depth",
        "focus": "Guidance Type",
        "page": "Page",
        "progress": "Progress",
        "source_type": "Source Type",
        "created_at": "Created",
        "last_opened": "Last opened",
        "no_subject_loaded": "No subject is currently loaded.",
        "no_subjects": "No saved subjects yet.",
        "status_new": "New",
        "status_in_progress": "In Progress",
        "status_completed": "Completed",
        "download_txt": "Download TXT",
        "download_json": "Download JSON",
        "waiting_content": "Waiting for learning content to appear.",
        "use_commands": "You can type naturally or use the Next button.",
        "title_fallback": "Untitled Subject",
        "knowledge_loaded": "ISO 19650 knowledge files loaded successfully.",
        "knowledge_missing": "ISO 19650 knowledge files were not found.",
        "focus_hint": "This helps the agent understand the kind of support needed before the chat starts.",
    },
}


# =========================================================
# General Helpers
# =========================================================

def now_iso() -> str:
    return datetime.now().isoformat()


def ensure_storage_dirs() -> None:
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    ISO_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines: List[str] = []
    blank_count = 0

    for line in lines:
        stripped = re.sub(r"[ \t]+", " ", line).strip()

        if not stripped:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
            continue

        blank_count = 0
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def safe_read_text(path: Path) -> str:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1256", "latin-1"]

    for encoding in encodings_to_try:
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue

    return ""


def format_dt(value: Optional[str]) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value


def infer_source_type(raw_text: str, uploaded_name: Optional[str] = None) -> str:
    name = (uploaded_name or "").lower()

    if name.endswith(".md"):
        return "markdown"
    if name.endswith(".txt"):
        return "text_file"

    lowered = raw_text.lower()

    if "iso 19650" in lowered:
        return "iso_standard"
    if "workflow" in lowered or "process" in lowered:
        return "workflow"
    if "bim" in lowered:
        return "bim_topic"
    if "```" in raw_text:
        return "code"

    return "pasted_text"


def infer_subject_title(raw_text: str, fallback_number: int, ui_lang: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    for line in lines[:12]:
        candidate = re.sub(r"^#{1,6}\s*", "", line)
        candidate = re.sub(r"^\d+[\.\-\)]\s*", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" -:|")

        if len(candidate) >= 5:
            return candidate[:100]

    if ui_lang == "ar":
        return f"موضوع {fallback_number:02d}"
    return f"Subject {fallback_number:02d}"


def get_ui_texts() -> Dict[str, str]:
    ui_lang = st.session_state.get("ui_language", "ar")
    return UI_TEXTS.get(ui_lang, UI_TEXTS["ar"])


def label_for_ui_language(code: str) -> str:
    return UI_LANGUAGE_OPTIONS.get(code, code)


def label_for_teaching_language(code: str) -> str:
    return TEACHING_LANGUAGE_OPTIONS.get(code, code)


def label_for_focus(code: str, ui_lang: str) -> str:
    return GUIDANCE_FOCUS_OPTIONS.get(ui_lang, GUIDANCE_FOCUS_OPTIONS["ar"]).get(code, code)


def label_for_depth(code: str, ui_lang: str) -> str:
    return LEARNING_DEPTH_OPTIONS.get(ui_lang, LEARNING_DEPTH_OPTIONS["ar"]).get(code, code)


# =========================================================
# ISO Knowledge Loading
# =========================================================

def normalize_for_search(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords_from_text(text: str, max_keywords: int = 20) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_/\-\.]{3,}", text.lower())

    stop_words = {
        "the", "and", "for", "with", "that", "from", "this", "into", "are", "was",
        "have", "has", "had", "shall", "should", "will", "would", "can", "could",
        "part", "iso", "19650", "information", "management"
    }

    freq: Dict[str, int] = {}
    for token in tokens:
        if token in stop_words:
            continue
        freq[token] = freq.get(token, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in ranked[:max_keywords]]


def load_iso_knowledge() -> Dict[str, Dict[str, Any]]:
    knowledge: Dict[str, Dict[str, Any]] = {}

    if not ISO_KNOWLEDGE_DIR.exists():
        return knowledge

    for path in sorted(ISO_KNOWLEDGE_DIR.glob("part_*.txt")):
        raw_text = safe_read_text(path)
        if not raw_text.strip():
            continue

        cleaned_text = normalize_whitespace(raw_text)
        match = re.search(r"part[_\-\s]?(\d+)", path.stem.lower())
        part_number = int(match.group(1)) if match else None

        knowledge[path.stem] = {
            "id": path.stem,
            "file_name": path.name,
            "part_number": part_number,
            "title": f"ISO 19650 Part {part_number}" if part_number else path.stem,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "search_text": normalize_for_search(cleaned_text),
            "keywords": extract_keywords_from_text(cleaned_text),
            "loaded_at": now_iso(),
        }

    return knowledge


def knowledge_is_available() -> bool:
    if not ISO_KNOWLEDGE_DIR.exists():
        return False
    return any(ISO_KNOWLEDGE_DIR.glob("part_*.txt"))


# =========================================================
# Session State
# =========================================================

def init_session_state() -> None:
    defaults = {
        "state": "awaiting_ui_language",
        "ui_language": "ar",
        "teaching_language": "ar",
        "guidance_focus": "iso_work_guidance",
        "depth": "guided",
        "current_subject": None,
        "subjects": {},
        "chat_history": [],
        "available_models": [],
        "selected_model": None,
        "api_key": "",
        "temperature": 0.3,
        "iso_knowledge": {},
        "knowledge_ready": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.ui_language not in UI_LANGUAGE_OPTIONS:
        st.session_state.ui_language = "ar"

    if st.session_state.teaching_language not in TEACHING_LANGUAGE_OPTIONS:
        st.session_state.teaching_language = "ar"

    if st.session_state.guidance_focus not in GUIDANCE_FOCUS_OPTIONS["ar"]:
        st.session_state.guidance_focus = "iso_work_guidance"

    if st.session_state.depth not in LEARNING_DEPTH_OPTIONS["ar"]:
        st.session_state.depth = "guided"


def load_knowledge_into_session() -> None:
    knowledge = load_iso_knowledge()
    st.session_state.iso_knowledge = knowledge
    st.session_state.knowledge_ready = bool(knowledge)


def init_rag_system() -> None:
    """Initialize the RAG system with existing knowledge."""
    try:
        api_key = st.session_state.get("api_key")
        if not api_key:
            st.warning("RAG system requires OpenRouter API key. Please set it in settings.")
            return

        # Initialize components
        ingester = DocumentIngester()
        chunker = TextChunker()
        vector_store = VectorStore(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        retriever = DocumentRetriever(vector_store)
        prompt_builder = RAGPromptBuilder()

        # Index existing ISO knowledge if not already done
        if not vector_store.load_collection("engineering_docs"):
            documents = ingester.ingest_directory(str(ISO_KNOWLEDGE_DIR))
            if documents:
                chunked_docs = chunker.chunk_documents(documents)
                vector_store.create_collection("engineering_docs")
                vector_store.add_documents(chunked_docs)

        # Store in session
        st.session_state.rag_ingester = ingester
        st.session_state.rag_chunker = chunker
        st.session_state.rag_store = vector_store
        st.session_state.rag_retriever = retriever
        st.session_state.rag_prompt_builder = prompt_builder
        st.session_state.rag_initialized = True

    except Exception as e:
        error_msg = str(e)
        if "402" in error_msg and "Insufficient credits" in error_msg:
            st.error("❌ **OpenRouter Credits Required**\n\nYour OpenRouter account needs credits to use the RAG system. Please:\n\n1. Visit [OpenRouter Credits](https://openrouter.ai/settings/credits)\n2. Purchase credits for your account\n3. The RAG system will work once credits are available")
        elif "401" in error_msg and "User not found" in error_msg:
            st.error("❌ **Invalid OpenRouter API Key**\n\nThe API key is not recognized. Please:\n\n1. Check your API key at [OpenRouter Keys](https://openrouter.ai/keys)\n2. Ensure you're using the correct account\n3. Generate a new key if needed")
        else:
            st.error(f"Failed to initialize RAG system: {e}")
        st.session_state.rag_initialized = False


# =========================================================
# RTL / Style Helpers
# =========================================================

def is_rtl_active() -> bool:
    ui_lang = st.session_state.get("ui_language", "ar")
    teaching_lang = st.session_state.get("teaching_language", "ar")
    return ui_lang == "ar" or teaching_lang == "ar"


def apply_app_styles() -> None:
    rtl = is_rtl_active()

    base_css = """
    <style>
    .stApp {
        background: transparent;
        color: inherit;
    }

    section[data-testid="stSidebar"] {
        background: transparent;
        border-right: 1px solid rgba(127, 127, 127, 0.18);
        padding: 0.75rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.2rem;
        max-width: 1250px;
    }

    h1, h2, h3, h4 {
        color: inherit;
        letter-spacing: 0;
    }

    h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    h2 {
        font-size: 1.35rem;
        font-weight: 600;
    }

    h3 {
        font-size: 1.1rem;
        font-weight: 600;
    }

    .stButton > button {
        width: 100%;
        min-height: 2.6rem;
        border-radius: 10px;
        border: 1px solid rgba(127, 127, 127, 0.25);
        background: transparent;
        color: inherit;
    }

    .stButton > button:hover {
        border-color: rgba(127, 127, 127, 0.45);
        background: rgba(127, 127, 127, 0.08);
    }

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border-radius: 10px;
    }

    .stSelectbox > div > div {
        border-radius: 10px;
    }

    .stExpander {
        border: 1px solid rgba(127, 127, 127, 0.16);
        border-radius: 10px;
    }

    .stChatMessage {
        border-bottom: 1px solid rgba(127, 127, 127, 0.12);
        padding-bottom: 0.7rem;
    }

    .app-card {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(127, 127, 127, 0.15);
        border-radius: 14px;
        margin-bottom: 0.75rem;
    }

    .meta-line {
        opacity: 0.82;
        font-size: 0.9rem;
    }

    .library-card {
        padding: 0.7rem 0.8rem;
        border: 1px solid rgba(127, 127, 127, 0.14);
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }
    """

    rtl_css = """
    .stApp,
    .stMarkdown,
    .stTextInput,
    .stTextArea,
    .stSelectbox,
    .stNumberInput,
    .stChatMessage,
    .stCaption,
    label,
    p,
    h1, h2, h3, h4, h5, h6,
    li,
    div[data-testid="stMarkdownContainer"] {
        direction: rtl;
        text-align: right;
    }
    """

    ltr_css = """
    .stApp,
    .stMarkdown,
    .stTextInput,
    .stTextArea,
    .stSelectbox,
    .stNumberInput,
    .stChatMessage,
    .stCaption,
    label,
    p,
    h1, h2, h3, h4, h5, h6,
    li,
    div[data-testid="stMarkdownContainer"] {
        direction: ltr;
        text-align: left;
    }
    """

    closing = "</style>"

    st.markdown(
        base_css + (rtl_css if rtl else ltr_css) + closing,
        unsafe_allow_html=True,
    )
    # =========================================================
# Subject / Lesson Data Helpers
# =========================================================

def clean_material_text(text: str) -> str:
    if not text:
        return ""
    return normalize_whitespace(text)


def get_next_subject_number(subjects: Dict[str, Dict[str, Any]]) -> int:
    numbers: List[int] = []

    for subject in subjects.values():
        number = subject.get("library_number")
        if isinstance(number, int):
            numbers.append(number)

    return max(numbers, default=0) + 1


def get_subject_status(subject: Dict[str, Any]) -> str:
    progress = int(subject.get("progress", 0) or 0)

    if progress <= 0:
        return "new"
    if progress >= 100:
        return "completed"
    return "in_progress"


def normalize_subject(subject: Dict[str, Any], ui_lang: str = "ar") -> Dict[str, Any]:
    subject = dict(subject)

    created_at = subject.get("created_at") or now_iso()

    subject.setdefault("id", str(uuid.uuid4()))
    subject.setdefault("library_number", 0)
    subject.setdefault("title", UI_TEXTS[ui_lang]["title_fallback"])
    subject.setdefault("source_type", "pasted_text")
    subject.setdefault("raw_content", "")
    subject.setdefault("cleaned_content", "")
    subject.setdefault("teaching_language", "ar")
    subject.setdefault("guidance_focus", "iso_work_guidance")
    subject.setdefault("depth", "guided")
    subject.setdefault("created_at", created_at)
    subject.setdefault("updated_at", created_at)
    subject.setdefault("last_opened_at", created_at)
    subject.setdefault("total_pages", None)
    subject.setdefault("current_page", 1)
    subject.setdefault("progress", 0)
    subject.setdefault("outline", "")
    subject.setdefault("generated_pages", [])
    subject.setdefault("chat_history", [])
    subject.setdefault("source_file_name", None)
    subject.setdefault("source_language", None)
    subject.setdefault("input_method", "pasted_text")
    subject.setdefault("relevant_iso_parts", [])
    subject.setdefault("page_size_mode", "long")

    return subject


def update_subject_progress(subject: Dict[str, Any]) -> Dict[str, Any]:
    total_pages = int(subject.get("total_pages") or 1)
    current_page = int(subject.get("current_page") or 1)

    completed_pages = max(0, min(current_page - 1, total_pages))
    subject["progress"] = min(100, int((completed_pages / total_pages) * 100))
    return subject


def get_generated_page_content(subject: Dict[str, Any], page_index: int) -> str:
    for item in subject.get("generated_pages", []) or []:
        if int(item.get("page_index", 0) or 0) == int(page_index):
            return item.get("content", "")
    return ""


def page_exists(subject: Dict[str, Any], page_index: int) -> bool:
    return bool(get_generated_page_content(subject, page_index).strip())


def append_generated_page(subject: Dict[str, Any], page_index: int, content: str) -> Dict[str, Any]:
    entry = {
        "page_index": int(page_index),
        "content": content,
        "generated_at": now_iso(),
    }

    subject.setdefault("generated_pages", [])
    existing_idx = int(page_index) - 1

    if 0 <= existing_idx < len(subject["generated_pages"]):
        subject["generated_pages"][existing_idx] = entry
    else:
        while len(subject["generated_pages"]) < existing_idx:
            subject["generated_pages"].append(
                {
                    "page_index": len(subject["generated_pages"]) + 1,
                    "content": "",
                    "generated_at": now_iso(),
                }
            )
        subject["generated_pages"].append(entry)

    subject["current_page"] = int(page_index)
    update_subject_progress(subject)
    save_subject(subject)
    return subject


# =========================================================
# Subject Storage
# =========================================================

def load_subjects() -> Dict[str, Dict[str, Any]]:
    ensure_storage_dirs()

    subjects: Dict[str, Dict[str, Any]] = {}
    ui_lang = st.session_state.get("ui_language", "ar")

    for path in sorted(SUBJECTS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                subject = json.load(f)

            subject = normalize_subject(subject, ui_lang=ui_lang)
            subjects[subject["id"]] = subject

        except Exception:
            continue

    return subjects


def save_subject(subject: Dict[str, Any]) -> None:
    ensure_storage_dirs()
    subject["updated_at"] = now_iso()

    file_path = SUBJECTS_DIR / f"{subject['id']}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(subject, f, ensure_ascii=False, indent=2)

    # Ingest subject material into RAG
    if st.session_state.get("rag_initialized") and subject.get("cleaned_content"):
        try:
            chunker = st.session_state.rag_chunker
            vector_store = st.session_state.rag_store

            # Create document from subject content
            doc_content = subject["cleaned_content"]
            metadata = {
                "subject_id": subject["id"],
                "subject_title": subject.get("title", ""),
                "source_type": "user_subject"
            }
            langchain_docs = chunker.chunk_text(doc_content, metadata)
            vector_store.add_documents(langchain_docs)
        except Exception as e:
            st.warning(f"Failed to index subject in RAG: {e}")


def delete_subject_file(subject_id: str) -> None:
    file_path = SUBJECTS_DIR / f"{subject_id}.json"
    if file_path.exists():
        file_path.unlink()


def load_subject_into_session(subject: Dict[str, Any]) -> Dict[str, Any]:
    subject["last_opened_at"] = now_iso()

    st.session_state.current_subject = subject
    st.session_state.chat_history = subject.get("chat_history", []) or []
    st.session_state.teaching_language = subject.get(
        "teaching_language",
        st.session_state.get("teaching_language", "ar")
    )
    st.session_state.guidance_focus = subject.get(
        "guidance_focus",
        st.session_state.get("guidance_focus", "iso_work_guidance")
    )
    st.session_state.depth = subject.get(
        "depth",
        st.session_state.get("depth", "guided")
    )

    save_subject(subject)
    return subject


def sync_subject_chat(subject: Dict[str, Any]) -> Dict[str, Any]:
    subject["chat_history"] = st.session_state.chat_history
    subject["last_opened_at"] = now_iso()
    update_subject_progress(subject)
    save_subject(subject)
    return subject


# =========================================================
# Subject Creation
# =========================================================

def subject_title_from_content(raw_text: str, fallback_number: int, ui_lang: str) -> str:
    return infer_subject_title(raw_text=raw_text, fallback_number=fallback_number, ui_lang=ui_lang)


def build_new_subject(
    raw_content: str,
    teaching_language: str,
    guidance_focus: str,
    depth: str,
    input_method: str,
    uploaded_name: Optional[str],
    subjects: Dict[str, Dict[str, Any]],
    ui_lang: str,
) -> Dict[str, Any]:
    cleaned_content = clean_material_text(raw_content)
    library_number = get_next_subject_number(subjects)

    subject = {
        "id": str(uuid.uuid4()),
        "library_number": library_number,
        "title": subject_title_from_content(
            raw_text=cleaned_content,
            fallback_number=library_number,
            ui_lang=ui_lang,
        ),
        "source_type": infer_source_type(cleaned_content, uploaded_name),
        "raw_content": raw_content,
        "cleaned_content": cleaned_content,
        "teaching_language": teaching_language,
        "guidance_focus": guidance_focus,
        "depth": depth,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "last_opened_at": now_iso(),
        "total_pages": None,
        "current_page": 1,
        "progress": 0,
        "outline": "",
        "generated_pages": [],
        "chat_history": [],
        "source_file_name": uploaded_name,
        "source_language": None,
        "input_method": input_method,
        "relevant_iso_parts": [],
        "page_size_mode": "long",
    }

    return normalize_subject(subject, ui_lang=ui_lang)


# =========================================================
# Subject Labels / Display Helpers
# =========================================================

def subject_button_label(subject: Dict[str, Any], ui_lang: str) -> str:
    number = int(subject.get("library_number") or 0)
    title = subject.get("title") or UI_TEXTS[ui_lang]["title_fallback"]
    return f"{number:02d}. {title}"


def label_for_status(status: str, ui_lang: str) -> str:
    texts = UI_TEXTS[ui_lang]

    mapping = {
        "new": texts["status_new"],
        "in_progress": texts["status_in_progress"],
        "completed": texts["status_completed"],
    }

    return mapping.get(status, status)


def get_sorted_subjects() -> List[Dict[str, Any]]:
    subjects = list(st.session_state.subjects.values())
    subjects.sort(
        key=lambda s: (
            int(s.get("library_number") or 0),
            s.get("created_at") or "",
        )
    )
    return subjects


# =========================================================
# Session Flow Helpers
# =========================================================

def start_new_subject_flow() -> None:
    st.session_state.state = "awaiting_ui_language"
    st.session_state.teaching_language = "ar"
    st.session_state.guidance_focus = "iso_work_guidance"
    st.session_state.depth = "guided"
    st.session_state.current_subject = None
    st.session_state.chat_history = []
    st.rerun()


def refresh_current_subject_reference() -> None:
    current = st.session_state.get("current_subject")
    if not current:
        return

    current_id = current.get("id")
    if current_id and current_id in st.session_state.subjects:
        st.session_state.current_subject = st.session_state.subjects[current_id]


def update_subject_runtime_settings(subject: Dict[str, Any]) -> None:
    subject["teaching_language"] = st.session_state.get("teaching_language", "ar")
    subject["guidance_focus"] = st.session_state.get("guidance_focus", "iso_work_guidance")
    subject["depth"] = st.session_state.get("depth", "guided")
    save_subject(subject)
    # =========================================================
# OpenRouter / Model Helpers
# =========================================================

def is_free_router_model(model: Dict[str, Any]) -> bool:
    model_id = str(model.get("id", "")).lower()
    if ":free" in model_id:
        return True

    pricing = model.get("pricing") or {}
    prompt_cost = pricing.get("prompt") or pricing.get("input") or pricing.get("prompt_usd")
    completion_cost = pricing.get("completion") or pricing.get("output") or pricing.get("completion_usd")

    def is_zero(value: Any) -> bool:
        if value is None:
            return False
        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return str(value).strip().lower() in {"0", "0.0", "free", "zero"}

    return is_zero(prompt_cost) and is_zero(completion_cost)


def get_free_models(api_key: str) -> List[Dict[str, Any]]:
    if not api_key.strip():
        return []

    try:
        response = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=15,
        )
        response.raise_for_status()

        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return []

        free_models = [
            model for model in models
            if isinstance(model, dict) and is_free_router_model(model)
        ]

        free_models.sort(key=lambda x: str(x.get("id", "")).lower())
        return free_models

    except requests.exceptions.RequestException:
        return []
    except Exception:
        return []


def build_openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key.strip(),
        base_url=OPENROUTER_BASE_URL,
    )


def streamed_text_chunks(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue

            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue

            if isinstance(delta, dict):
                content = delta.get("content")
            else:
                content = getattr(delta, "content", None)

            if content:
                yield content

    except Exception as e:
        yield f"\n\n[Generation error] {str(e)}"


def buffered_stream_text(response_iterator):
    buffer = ""
    final_text = ""

    for token in response_iterator:
        buffer += token

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            final_text += line + "\n"
            yield final_text

        if buffer:
            yield final_text + buffer

    if buffer:
        final_text += buffer
        yield final_text


# =========================================================
# ISO Knowledge Retrieval Helpers
# =========================================================

def split_text_into_chunks(text: str, max_chars: int = 1800) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n\n" + paragraph

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                piece = paragraph[start:start + max_chars]
                chunks.append(piece.strip())
                start += max_chars
            current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_iso_chunk_index() -> List[Dict[str, Any]]:
    knowledge = st.session_state.get("iso_knowledge", {}) or {}
    index: List[Dict[str, Any]] = []

    for item in knowledge.values():
        part_number = item.get("part_number")
        title = item.get("title", "")
        cleaned_text = item.get("cleaned_text", "")

        for i, chunk in enumerate(split_text_into_chunks(cleaned_text, max_chars=1800), start=1):
            search_text = normalize_for_search(chunk)
            keywords = extract_keywords_from_text(chunk, max_keywords=15)

            index.append({
                "source_id": item.get("id"),
                "part_number": part_number,
                "title": title,
                "chunk_index": i,
                "chunk_text": chunk,
                "search_text": search_text,
                "keywords": keywords,
            })

    return index


def tokenize_for_match(text: str) -> List[str]:
    english_tokens = re.findall(r"[A-Za-z0-9_/\-\.]{3,}", text.lower())

    arabic_tokens = re.findall(r"[\u0600-\u06FF]{2,}", text)
    arabic_tokens = [token.strip() for token in arabic_tokens if len(token.strip()) >= 2]

    tokens = english_tokens + arabic_tokens

    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "have", "has", "had",
        "are", "was", "were", "what", "when", "where", "which", "shall", "should", "will",
        "would", "could", "about", "work", "page", "pages", "part", "parts",
        "على", "في", "من", "الى", "إلى", "عن", "ما", "ماذا", "كيف", "هل", "هذا", "هذه",
        "هناك", "عند", "شرح", "موضوع", "صفحة", "صفحات", "جزء", "أجزاء"
    }

    unique_tokens: List[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in stop_words:
            continue
        if normalized not in unique_tokens:
            unique_tokens.append(normalized)

    return unique_tokens


def score_iso_chunk(query_tokens: List[str], chunk: Dict[str, Any]) -> int:
    if not query_tokens:
        return 0

    search_text = chunk.get("search_text", "")
    keywords = chunk.get("keywords", []) or []

    score = 0

    for token in query_tokens:
        if token in search_text:
            score += 3
        if token in keywords:
            score += 5

    part_number = chunk.get("part_number")
    if part_number in {1, 2, 3, 4, 5}:
        score += 1

    return score


def detect_relevant_iso_parts_from_text(text: str) -> List[int]:
    lowered = text.lower()
    matched_parts: List[int] = []

    manual_rules = {
        1: ["principles", "concepts", "concept", "definitions", "overview", "framework", "مبادئ", "مفاهيم", "تعريفات"],
        2: ["delivery phase", "delivery", "project delivery", "appointing party", "lead appointed party", "capital phase", "مرحلة التسليم", "التسليم", "التعيين"],
        3: ["operational phase", "operations", "asset", "facility management", "operation", "التشغيل", "الأصول", "إدارة المرافق"],
        4: ["information exchange", "exchange", "delivery cycle", "information delivery", "تبادل المعلومات", "تسليم المعلومات"],
        5: ["security", "secure", "sensitive", "cyber", "security-minded", "الأمن", "حماية", "حساس", "أمن المعلومات"],
    }

    for part_number, terms in manual_rules.items():
        if any(term in lowered for term in terms):
            matched_parts.append(part_number)

    return matched_parts


def build_search_query_from_subject(subject: Dict[str, Any], extra_text: str = "") -> str:
    components = [
        subject.get("title", ""),
        subject.get("cleaned_content", "")[:2500],
        subject.get("source_type", ""),
        subject.get("guidance_focus", ""),
        extra_text,
    ]
    return "\n".join([c for c in components if c]).strip()


def retrieve_relevant_iso_chunks(
    subject: Dict[str, Any],
    extra_text: str = "",
    max_chunks: int = 6,
) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks using RAG system."""
    if not st.session_state.get("rag_initialized"):
        return []

    query_text = build_search_query_from_subject(subject, extra_text=extra_text)

    try:
        retriever = st.session_state.rag_retriever
        retrieved_docs = retriever.retrieve(query_text, k=max_chunks)

        # Convert to expected format
        chunks = []
        for i, doc in enumerate(retrieved_docs):
            # Extract part number from metadata or filename
            part_number = None
            filename = doc.metadata.get('filename', '')
            match = re.search(r"part[_\-\s]?(\d+)", filename.lower())
            if match:
                part_number = int(match.group(1))

            chunks.append({
                "part_number": part_number,
                "chunk_index": i + 1,
                "chunk_text": doc.page_content,
                "score": 1.0,  # Simplified scoring
            })

        return chunks

    except Exception as e:
        st.error(f"RAG retrieval failed: {e}")
        return []


def build_iso_context_block(subject: Dict[str, Any], extra_text: str = "") -> str:
    relevant_chunks = retrieve_relevant_iso_chunks(subject, extra_text=extra_text, max_chunks=6)

    subject["relevant_iso_parts"] = sorted(
        list({int(item.get("part_number")) for item in relevant_chunks if item.get("part_number")})
    )

    if not relevant_chunks:
        return "No ISO 19650 knowledge context was matched."

    blocks: List[str] = []
    for item in relevant_chunks:
        part_number = item.get("part_number", "?")
        chunk_index = item.get("chunk_index", "?")
        chunk_text = item.get("chunk_text", "")
        blocks.append(
            f"[ISO 19650 Part {part_number} | Chunk {chunk_index}]\n{chunk_text}"
        )

    return "\n\n".join(blocks).strip()


def get_iso_knowledge_status_message() -> str:
    texts = get_ui_texts()
    if st.session_state.get("knowledge_ready"):
        return texts["knowledge_loaded"]
    return texts["knowledge_missing"]


# =========================================================
# Message / Generation Helpers
# =========================================================

def generated_pages_text(subject: Dict[str, Any]) -> str:
    pages = subject.get("generated_pages", []) or []
    if not pages:
        return "No pages have been generated yet."

    blocks: List[str] = []
    for item in pages:
        page_index = item.get("page_index", "?")
        content = item.get("content", "")
        blocks.append(f"Page {page_index}:\n{content}")

    return "\n\n".join(blocks)


def generate_streamed_markdown(
    messages: List[Dict[str, str]],
    client: OpenAI,
    model: str,
    temperature: float,
    placeholder,
) -> str:
    final_text = ""

    iterator = buffered_stream_text(
        streamed_text_chunks(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
        )
    )

    for rendered in iterator:
        final_text = rendered
        placeholder.markdown(final_text)

    return final_text.strip()
# =========================================================
# Instructor Engine
# =========================================================

def subject_context_block(subject: Dict[str, Any], page_index: Optional[int] = None) -> str:
    return f"""
Current subject title: {subject.get('title', 'Unknown')}
Source type: {subject.get('source_type', 'unknown')}
Teaching language: {subject.get('teaching_language', 'ar')}
Guidance focus: {subject.get('guidance_focus', 'iso_work_guidance')}
Learning depth: {subject.get('depth', 'guided')}
Current page: {page_index or subject.get('current_page', 1)}
Total pages: {subject.get('total_pages', 'Unknown')}
Relevant ISO parts: {subject.get('relevant_iso_parts', [])}
Saved outline: {subject.get('outline', '') or 'Not generated yet'}
Already generated pages count: {len(subject.get('generated_pages', []))}
"""


def build_identity_layer(subject: Dict[str, Any]) -> str:
    language = subject.get("teaching_language", "ar")
    focus = subject.get("guidance_focus", "iso_work_guidance")

    arabic_identity = """
You are ISO19650 Agent.

You are not a generic chatbot.
You are a structured teaching and work-guidance agent.

Your behavior rules:
- If the user asks about work-related topics, BIM workflows, documentation logic, information management, roles, exchanges, delivery, CDE, naming, approvals, responsibilities, or coordination, answer in a way aligned with ISO 19650 whenever relevant.
- Keep the Instructor Companion mentality:
  explain step by step,
  build understanding progressively,
  clarify confusion points,
  teach like a calm professional instructor.
- When Arabic is used, write in clear natural Arabic with a direct instructional tone.
- Avoid stiff formal Arabic unless truly necessary.
- Do not sound playful or generic.
- Do not use mini recap sections.
- Do not add “what comes next” teaser sections.
- Do not keep the response too short.
- Each generated page should feel like a substantial teaching page, not a brief note.
"""

    english_identity = """
You are ISO19650 Agent.

You are not a generic chatbot.
You are a structured teaching and work-guidance agent.

Your behavior rules:
- If the user asks about work-related topics, BIM workflows, documentation logic, information management, roles, exchanges, delivery, CDE, naming, approvals, responsibilities, or coordination, answer in a way aligned with ISO 19650 whenever relevant.
- Keep the Instructor Companion mentality:
  explain step by step,
  build understanding progressively,
  clarify confusion points,
  teach like a calm professional instructor.
- Use clear, professional, readable English.
- Do not sound playful or generic.
- Do not use mini recap sections.
- Do not add “what comes next” teaser sections.
- Do not keep the response too short.
- Each generated page should feel like a substantial teaching page, not a brief note.
"""

    focus_layer = {
        "iso_work_guidance": """
Special mode: ISO 19650 Work Guidance
- Prioritize ISO 19650 interpretation for work situations.
- If the user asks “how should we do this at work?”, answer according to ISO 19650 principles when applicable.
- Make the answer practical, not purely theoretical.
""",
        "bim_workflow_guidance": """
Special mode: BIM Workflow Guidance
- Explain BIM workflows step by step.
- Bring ISO 19650 alignment when relevant, but keep the answer practical and workflow-oriented.
""",
        "problem_explanation": """
Special mode: Problem Explanation
- The user may be stuck, confused, or facing an issue.
- First explain the root idea clearly.
- Then explain where the confusion usually happens.
- Then explain the correct understanding step by step.
""",
        "general_learning": """
Special mode: General Learning
- Explain the material progressively and structurally.
- Use the teaching mindset strongly even if the topic is broader than ISO 19650.
""",
    }

    base = arabic_identity if language == "ar" else english_identity
    return base + "\n" + focus_layer.get(focus, focus_layer["general_learning"])


def build_state_layer(state: str) -> str:
    return f"""
Current app state: {state}

State rules:
- awaiting_material: do not teach yet
- awaiting_pages: do not start page generation yet
- teaching: generate only the requested page or requested teaching action
- reviewing: generate a structured review
- summarizing: generate a useful summary
"""


def build_pedagogy_layer(subject: Dict[str, Any]) -> str:
    focus = subject.get("guidance_focus", "general_learning")

    base = """
Teaching protocol:
- Start from why the concept matters before jumping to the formal wording, when appropriate.
- Build each idea on what comes before it.
- Explain important terms at first appearance.
- Do not stack unexplained terms.
- Use headings and clean sectioning when helpful.
- Use examples when they improve understanding.
- Maintain flow and coherence across pages.

Important restrictions:
- Do not add a dedicated mini recap section.
- Do not add a “next page” transition section.
- Do not make the answer overly compressed.
- Avoid short shallow replies.
"""

    if focus == "iso_work_guidance":
        extra = """
Work-guidance mode:
- When relevant, connect the answer to ISO 19650 principles, roles, information requirements, delivery logic, CDE logic, responsibilities, approvals, and information flow.
- Make the answer usable for real work situations.
- Distinguish between principle, process, and practical application.
"""
    elif focus == "bim_workflow_guidance":
        extra = """
BIM workflow mode:
- Explain what the workflow is trying to achieve.
- Explain sequence, dependencies, roles, expected outputs, and common workflow confusion points.
- Connect to ISO 19650 only when relevant.
"""
    elif focus == "problem_explanation":
        extra = """
Problem explanation mode:
- Clarify the misunderstanding first.
- Then explain the correct idea step by step.
- Focus on where people usually get confused.
"""
    else:
        extra = """
General learning mode:
- Keep the explanation structured, progressive, and educational.
"""

    return base + "\n" + extra


def build_depth_layer(subject: Dict[str, Any]) -> str:
    depth = subject.get("depth", "guided")

    if depth == "quick":
        return """
Depth mode = Quick Overview
- keep it shorter than other modes
- still meaningful and complete
- focus on the main idea and practical understanding
- do not become too brief
"""
    if depth == "deep":
        return """
Depth mode = Deep Learning
- explain with more expansion
- unpack terms more carefully
- make relations between ideas more explicit
- use fuller explanations and more supporting detail
- the output should be noticeably longer
"""
    return """
Depth mode = Guided Learning
- balanced explanation
- clear progression
- enough development to feel like a real teaching page
- not too short
"""


def build_output_constraints(action: str, page_index: Optional[int] = None) -> str:
    if action == "outline":
        return """
Output task:
Generate a practical internal outline for the material.
Return a clean structured outline only.
Do not return filler text.
"""

    if action == "page":
        return f"""
Output task:
Generate only Page {page_index}.

Page writing requirements:
- It must feel like one real teaching page.
- It must not be too short.
- Use meaningful section development.
- Explain the idea properly.
- Do not generate other pages.
- Do not add a mini recap section.
- Do not add a “what comes next” section.
"""

    if action == "summary":
        return """
Output task:
Generate a useful summary of the lesson progress so far.
Do not restart from zero.
Do not make it too short.
"""

    if action == "review":
        return """
Output task:
Generate a structured review of what has already been explained.
Focus on consolidation and clarification.
"""

    if action == "simpler":
        return """
Output task:
Re-explain the current page in a simpler and clearer way.
Keep it substantial.
"""

    if action == "deeper":
        return """
Output task:
Expand the current page with deeper explanation and stronger concept development.
Make it noticeably fuller.
"""

    return """
Output task:
Respond in the correct structured teaching mode.
"""


def build_prompt_layers(
    state: str,
    subject: Dict[str, Any],
    action: str,
    page_index: Optional[int] = None,
    extra_text: str = "",
) -> str:
    iso_context = build_iso_context_block(subject, extra_text=extra_text)

    return "\n".join(
        [
            build_identity_layer(subject),
            build_state_layer(state),
            subject_context_block(subject, page_index=page_index),
            build_pedagogy_layer(subject),
            build_depth_layer(subject),
            build_output_constraints(action, page_index=page_index),
            "ISO 19650 reference context:",
            iso_context,
        ]
    )


# =========================================================
# Prompt Builders
# =========================================================

def build_outline_messages(subject: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="teaching",
        subject=subject,
        action="outline",
        extra_text=subject.get("cleaned_content", ""),
    )

    user_content = f"""
Create an internal teaching outline for the following material.

Source material:
{subject.get('cleaned_content') or subject.get('raw_content') or ''}
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_page_messages(subject: Dict[str, Any], page_index: int) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="teaching",
        subject=subject,
        action="page",
        page_index=page_index,
        extra_text=(subject.get("cleaned_content", "")[:2500] + "\n" + generated_pages_text(subject)[:2500]),
    )

    user_content = f"""
Generate Page {page_index} only.

Source material:
{subject.get('cleaned_content') or subject.get('raw_content') or ''}

Already generated pages:
{generated_pages_text(subject)}

Important:
- Generate only the requested page.
- Keep continuity with earlier pages if they exist.
- Make the page substantial, not short.
- No mini recap section.
- No transition section to the next page.
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_summary_messages(subject: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="summarizing",
        subject=subject,
        action="summary",
        extra_text=generated_pages_text(subject),
    )

    user_content = f"""
Summarize the lesson progress so far based on the generated pages below.

Generated pages:
{generated_pages_text(subject)}
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_review_messages(subject: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="reviewing",
        subject=subject,
        action="review",
        extra_text=generated_pages_text(subject),
    )

    user_content = f"""
Create a structured review of what has already been explained.

Generated pages:
{generated_pages_text(subject)}
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_simpler_messages(subject: Dict[str, Any], current_page: int) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="teaching",
        subject=subject,
        action="simpler",
        page_index=current_page,
        extra_text=get_generated_page_content(subject, current_page),
    )

    user_content = f"""
Re-explain Page {current_page} in a simpler and clearer way.

Current page content:
{get_generated_page_content(subject, current_page)}

Source material:
{subject.get('cleaned_content') or subject.get('raw_content') or ''}
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def build_deeper_messages(subject: Dict[str, Any], current_page: int) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="teaching",
        subject=subject,
        action="deeper",
        page_index=current_page,
        extra_text=get_generated_page_content(subject, current_page),
    )

    user_content = f"""
Expand Page {current_page} with a deeper explanation.

Current page content:
{get_generated_page_content(subject, current_page)}

Source material:
{subject.get('cleaned_content') or subject.get('raw_content') or ''}
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


# =========================================================
# Generation Actions
# =========================================================

def ensure_outline_generated(
    subject: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
) -> None:
    if subject.get("outline"):
        return

    messages = build_outline_messages(subject)
    parts: List[str] = []

    for token in streamed_text_chunks(client, messages, model, temperature):
        parts.append(token)

    outline = "".join(parts).strip()
    if outline:
        subject["outline"] = outline
        save_subject(subject)


def generate_page_and_update(
    subject: Dict[str, Any],
    page_index: int,
    client: OpenAI,
    model: str,
    temperature: float,
) -> str:
    ensure_outline_generated(subject, client, model, temperature)

    placeholder = st.empty()
    messages = build_page_messages(subject, page_index)

    full_text = generate_streamed_markdown(
        messages=messages,
        client=client,
        model=model,
        temperature=temperature,
        placeholder=placeholder,
    )

    if full_text:
        append_generated_page(subject, page_index, full_text)
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        sync_subject_chat(subject)

    return full_text


def generate_summary_response(
    subject: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
) -> str:
    placeholder = st.empty()
    messages = build_summary_messages(subject)

    full_text = generate_streamed_markdown(
        messages=messages,
        client=client,
        model=model,
        temperature=temperature,
        placeholder=placeholder,
    )

    if full_text:
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        sync_subject_chat(subject)

    return full_text


def generate_review_response(
    subject: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
) -> str:
    placeholder = st.empty()
    messages = build_review_messages(subject)

    full_text = generate_streamed_markdown(
        messages=messages,
        client=client,
        model=model,
        temperature=temperature,
        placeholder=placeholder,
    )

    if full_text:
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        sync_subject_chat(subject)

    return full_text


def regenerate_current_page_simpler(
    subject: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
) -> str:
    current_page = int(subject.get("current_page") or 1)
    placeholder = st.empty()

    full_text = generate_streamed_markdown(
        messages=build_simpler_messages(subject, current_page),
        client=client,
        model=model,
        temperature=temperature,
        placeholder=placeholder,
    )

    if full_text:
        append_generated_page(subject, current_page, full_text)
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        sync_subject_chat(subject)

    return full_text


def regenerate_current_page_deeper(
    subject: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
) -> str:
    current_page = int(subject.get("current_page") or 1)
    placeholder = st.empty()

    full_text = generate_streamed_markdown(
        messages=build_deeper_messages(subject, current_page),
        client=client,
        model=model,
        temperature=temperature,
        placeholder=placeholder,
    )

    if full_text:
        append_generated_page(subject, current_page, full_text)
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        sync_subject_chat(subject)

    return full_text
# =========================================================
# Command Interpretation
# =========================================================

def normalize_user_input(text: str) -> str:
    if not text:
        return ""
    return normalize_whitespace(text).strip()


def interpret_user_command(user_input: str) -> Dict[str, Any]:
    raw = normalize_user_input(user_input)
    lowered = raw.lower()

    if not raw:
        return {"action": "empty"}

    if raw.isdigit():
        return {"action": "jump_page", "page_index": int(raw)}

    next_terms = {
        "next", "continue", "go next", "next page",
        "كمل", "التالي", "الصفحة التالية", "كمّل", "نكمل", "بعد كده"
    }
    if lowered in next_terms:
        return {"action": "next_page"}

    simpler_terms = [
        "simpler", "simplify", "explain simply", "make it simpler",
        "أبسط", "بسطها", "اشرحها أبسط", "بسّطها", "وضحها أبسط"
    ]
    if any(term in lowered for term in simpler_terms):
        return {"action": "simpler"}

    deeper_terms = [
        "deeper", "expand", "more detail", "explain more", "go deeper",
        "أعمق", "وسع", "اشرح أكتر", "تفصيل أكتر", "فصل أكتر", "زود التفاصيل"
    ]
    if any(term in lowered for term in deeper_terms):
        return {"action": "deeper"}

    summary_terms = [
        "summary", "summarize", "summarise", "give me a summary",
        "لخص", "ملخص", "اعمل ملخص", "عايز ملخص"
    ]
    if any(term in lowered for term in summary_terms):
        return {"action": "summary"}

    review_terms = [
        "review", "recap", "revise",
        "راجع", "مراجعة", "اعمل مراجعة", "راجع معايا"
    ]
    if any(term in lowered for term in review_terms):
        return {"action": "review"}

    return {"action": "chat"}


def process_user_input(
    user_input: str,
    state: str,
    subject: Optional[Dict[str, Any]],
    texts: Dict[str, str],
) -> Dict[str, Any]:
    raw = normalize_user_input(user_input)

    if state == "awaiting_material":
        return {
            "next_state": "awaiting_pages",
            "message": texts["pages_question"],
            "action": "message_only",
        }

    if state == "awaiting_pages":

        try:
            pages = int(raw)
            if pages < 1:
                raise ValueError("Pages must be >= 1")
            if pages > MAX_PAGES:
                return {
                    "next_state": state,
                    "message": texts["max_pages_error"],
                    "action": "message_only",
                }

            return {
                "next_state": "teaching",
                "message": f"{texts['processing_input']} {pages}",
                "pages": pages,
                "action": "set_pages",
            }
        except Exception:
            return {
                "next_state": state,
                "message": texts["pages_question"],
                "action": "message_only",
            }

    if state in {"teaching", "reviewing", "summarizing"} and subject:
        interpreted = interpret_user_command(user_input)
        action = interpreted["action"]

        if action == "next_page":
            next_page = min(
                int(subject.get("total_pages") or 1),
                int(subject.get("current_page") or 1) + 1,
            )
            return {
                "next_state": "teaching",
                "action": "generate_page",
                "page_index": next_page,
                "message": texts["processing_input"],
            }

        if action == "jump_page":
            requested = int(interpreted.get("page_index", 1))
            total_pages = int(subject.get("total_pages") or 1)
            requested = max(1, min(requested, total_pages))

            return {
                "next_state": "teaching",
                "action": "generate_page",
                "page_index": requested,
                "message": texts["processing_input"],
            }

        if action == "simpler":
            return {
                "next_state": "teaching",
                "action": "simpler",
                "message": texts["processing_input"],
            }

        if action == "deeper":
            return {
                "next_state": "teaching",
                "action": "deeper",
                "message": texts["processing_input"],
            }

        if action == "summary":
            return {
                "next_state": "summarizing",
                "action": "summary",
                "message": texts["processing_input"],
            }

        if action == "review":
            return {
                "next_state": "reviewing",
                "action": "review",
                "message": texts["processing_input"],
            }

        if action == "empty":
            return {
                "next_state": state,
                "action": "message_only",
                "message": texts["use_commands"],
            }

        return {
            "next_state": state,
            "action": "free_chat",
            "message": raw,
        }

    return {
        "next_state": state,
        "action": "message_only",
        "message": texts["processing_input"],
    }


# =========================================================
# Free Chat / Explanation Message Builders
# =========================================================

def build_free_chat_messages(subject: Dict[str, Any], user_input: str) -> List[Dict[str, str]]:
    prompt = build_prompt_layers(
        state="teaching",
        subject=subject,
        action="page",
        page_index=subject.get("current_page", 1),
        extra_text=user_input,
    )

    user_content = f"""
The user is asking a direct follow-up or clarification.

User message:
{user_input}

Source material:
{subject.get('cleaned_content') or subject.get('raw_content') or ''}

Already generated pages:
{generated_pages_text(subject)}

Instructions:
- Answer the user's exact question.
- Keep the Instructor Companion mentality.
- If this is work-related, align with ISO 19650 when relevant.
- If this is a confusion/problem, explain the issue clearly and step by step.
- Do not make the answer too short.
- Do not add a mini recap section.
- Do not add a next-page teaser.
"""

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]


def generate_free_chat_response(
    subject: Dict[str, Any],
    user_input: str,
    client: OpenAI,
    model: str,
    temperature: float,
) -> str:
    placeholder = st.empty()
    messages = build_free_chat_messages(subject, user_input)

    full_text = generate_streamed_markdown(
        messages=messages,
        client=client,
        model=model,
        temperature=temperature,
        placeholder=placeholder,
    )

    if full_text:
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})
        sync_subject_chat(subject)

    return full_text


# =========================================================
# Export Helpers
# =========================================================

def build_export_text(subject: Dict[str, Any], chat_history: List[Dict[str, str]], ui_lang: str) -> str:
    texts = UI_TEXTS.get(ui_lang, UI_TEXTS["ar"])

    lines = [
        f"{texts['subject']}: {subject.get('title', '')}",
        f"{texts['language']}: {label_for_teaching_language(subject.get('teaching_language', 'ar'))}",
        f"{texts['focus']}: {label_for_focus(subject.get('guidance_focus', 'general_learning'), ui_lang)}",
        f"{texts['depth']}: {label_for_depth(subject.get('depth', 'guided'), ui_lang)}",
        f"{texts['progress']}: {subject.get('progress', 0)}%",
        f"{texts['source_type']}: {subject.get('source_type', '')}",
        f"{texts['created_at']}: {format_dt(subject.get('created_at'))}",
        f"{texts['last_opened']}: {format_dt(subject.get('last_opened_at'))}",
        "",
        "====================",
        "CHAT HISTORY",
        "====================",
        "",
    ]

    for msg in chat_history:
        role = str(msg.get("role", "assistant")).upper()
        content = msg.get("content", "")
        lines.append(f"{role}:")
        lines.append(content)
        lines.append("")

    return "\n".join(lines).strip()


def build_export_json(subject: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    payload = {
        "subject": subject,
        "chat_history": chat_history,
        "exported_at": now_iso(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_export_actions(subject: Dict[str, Any]) -> None:
    ui_lang = st.session_state.get("ui_language", "ar")
    texts = UI_TEXTS.get(ui_lang, UI_TEXTS["ar"])

    export_txt = build_export_text(subject, st.session_state.chat_history, ui_lang)
    export_json = build_export_json(subject, st.session_state.chat_history)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            texts["download_txt"],
            data=export_txt,
            file_name=f"{subject.get('title', 'lesson').replace(' ', '_')}.txt",
            mime="text/plain",
            key="download_txt_button",
        )

    with col2:
        st.download_button(
            texts["download_json"],
            data=export_json,
            file_name=f"{subject.get('title', 'lesson').replace(' ', '_')}.json",
            mime="application/json",
            key="download_json_button",
        )
        # =========================================================
# Header / Sidebar / Runtime Settings
# =========================================================

def render_header() -> None:
    texts = get_ui_texts()
    ui_lang = st.session_state.get("ui_language", "ar")

    st.title(APP_TITLE)

    subject = st.session_state.get("current_subject")
    if not subject:
        st.caption(get_iso_knowledge_status_message())
        return

    current_page = int(subject.get("current_page") or 1)
    total_pages = subject.get("total_pages") or "-"
    progress = int(subject.get("progress") or 0)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader(subject.get("title") or texts["title_fallback"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption(f"{texts['language']}: {label_for_teaching_language(subject.get('teaching_language', 'ar'))}")
    with c2:
        st.caption(f"{texts['focus']}: {label_for_focus(subject.get('guidance_focus', 'general_learning'), ui_lang)}")
    with c3:
        st.caption(f"{texts['depth']}: {label_for_depth(subject.get('depth', 'guided'), ui_lang)}")
    with c4:
        st.caption(f"{texts['page']}: {current_page}/{total_pages} · {texts['progress']}: {progress}%")

    st.caption(get_iso_knowledge_status_message())
    st.markdown("</div>", unsafe_allow_html=True)


def render_subject_snapshot(subject: Dict[str, Any]) -> None:
    texts = get_ui_texts()
    ui_lang = st.session_state.get("ui_language", "ar")

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown(f"**{texts['subject']}:** {subject.get('title') or texts['title_fallback']}")
    st.markdown(
        f"<div class='meta-line'>"
        f"{texts['source_type']}: {subject.get('source_type', '-')}"
        f" · {texts['created_at']}: {format_dt(subject.get('created_at'))}"
        f" · {texts['last_opened']}: {format_dt(subject.get('last_opened_at'))}"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"{texts['language']}: {label_for_teaching_language(subject.get('teaching_language', 'ar'))}")
    with c2:
        st.caption(f"{texts['focus']}: {label_for_focus(subject.get('guidance_focus', 'general_learning'), ui_lang)}")
    with c3:
        st.caption(f"{texts['depth']}: {label_for_depth(subject.get('depth', 'guided'), ui_lang)}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_left_panel() -> None:
    texts = get_ui_texts()
    ui_lang = st.session_state.get("ui_language", "ar")

    with st.sidebar:
        st.header(texts["subject_library"])

        if st.button(texts["start_new"], key="start_new_subject_btn"):
            start_new_subject_flow()

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        subjects = get_sorted_subjects()
        if not subjects:
            st.caption(texts["no_subjects"])
            return

        for subject in subjects:
            subject_id = subject["id"]
            status = get_subject_status(subject)
            current_page = int(subject.get("current_page") or 1)
            total_pages = subject.get("total_pages") or "-"
            label = subject_button_label(subject, ui_lang)

            st.markdown('<div class="library-card">', unsafe_allow_html=True)
            st.markdown(f"**{label}**")
            st.caption(
                f"{label_for_status(status, ui_lang)}"
                f" · {texts['page']}: {current_page}/{total_pages}"
                f" · {texts['progress']}: {int(subject.get('progress', 0) or 0)}%"
            )

            if st.button(f"{texts['resume']} · {label}", key=f"resume_{subject_id}"):
                st.session_state.subjects[subject_id] = load_subject_into_session(subject)
                st.session_state.state = "teaching" if subject.get("total_pages") else "awaiting_pages"
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


def render_runtime_settings(subject: Dict[str, Any]) -> Optional[OpenAI]:
    texts = get_ui_texts()

    with st.expander(texts["settings"], expanded=False):
        ui_language_codes = list(UI_LANGUAGE_OPTIONS.keys())
        current_ui_language = st.session_state.get("ui_language", "ar")
        if current_ui_language not in ui_language_codes:
            current_ui_language = "ar"

        selected_ui_language = st.selectbox(
            texts["ui_language"],
            options=ui_language_codes,
            index=ui_language_codes.index(current_ui_language),
            format_func=lambda x: label_for_ui_language(x),
            key="runtime_ui_language_select",
        )

        teaching_language_codes = list(TEACHING_LANGUAGE_OPTIONS.keys())
        current_teaching_language = st.session_state.get("teaching_language", subject.get("teaching_language", "ar"))
        if current_teaching_language not in teaching_language_codes:
            current_teaching_language = "ar"

        selected_teaching_language = st.selectbox(
            texts["teaching_language"],
            options=teaching_language_codes,
            index=teaching_language_codes.index(current_teaching_language),
            format_func=lambda x: label_for_teaching_language(x),
            key="runtime_teaching_language_select",
        )

        ui_lang_for_labels = selected_ui_language if selected_ui_language in GUIDANCE_FOCUS_OPTIONS else "ar"

        focus_codes = list(GUIDANCE_FOCUS_OPTIONS[ui_lang_for_labels].keys())
        current_focus = st.session_state.get("guidance_focus", subject.get("guidance_focus", "iso_work_guidance"))
        if current_focus not in focus_codes:
            current_focus = "iso_work_guidance"

        selected_focus = st.selectbox(
            texts["select_focus"],
            options=focus_codes,
            index=focus_codes.index(current_focus),
            format_func=lambda x: label_for_focus(x, ui_lang_for_labels),
            key="runtime_focus_select",
        )

        depth_codes = list(LEARNING_DEPTH_OPTIONS[ui_lang_for_labels].keys())
        current_depth = st.session_state.get("depth", subject.get("depth", "guided"))
        if current_depth not in depth_codes:
            current_depth = "guided"

        selected_depth = st.selectbox(
            texts["select_depth"],
            options=depth_codes,
            index=depth_codes.index(current_depth),
            format_func=lambda x: label_for_depth(x, ui_lang_for_labels),
            key="runtime_depth_select",
        )

        st.caption(texts["focus_hint"])

        st.session_state.ui_language = selected_ui_language
        st.session_state.teaching_language = selected_teaching_language
        st.session_state.guidance_focus = selected_focus
        st.session_state.depth = selected_depth

        update_subject_runtime_settings(subject)
        st.session_state.subjects[subject["id"]] = subject

        api_key_input = st.text_input(
            texts["api_key"],
            value=st.session_state.get("api_key", ""),
            type="password",
            key="runtime_api_key_input",
        )

        if api_key_input != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_key_input
            st.session_state.available_models = []
            st.session_state.selected_model = None
            # Re-initialize RAG system with new API key
            st.session_state.rag_initialized = False
            if api_key_input:
                init_rag_system()

        st.markdown(f"[{texts['get_key']}]({OPENROUTER_KEYS_URL})")

        if st.session_state.get("api_key", "").strip():
            if not st.session_state.get("available_models"):
                st.session_state.available_models = get_free_models(st.session_state["api_key"])

            if st.session_state.available_models:
                model_ids = [m.get("id", "") for m in st.session_state.available_models if m.get("id")]
                if model_ids:
                    current_model = st.session_state.get("selected_model")
                    if current_model not in model_ids:
                        current_model = model_ids[0]

                    st.session_state.selected_model = st.selectbox(
                        texts["select_model"],
                        options=model_ids,
                        index=model_ids.index(current_model),
                        key="runtime_model_select",
                    )
            else:
                st.warning(texts["no_models"])

        st.session_state.temperature = st.slider(
            texts["temperature"],
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("temperature", 0.3)),
            step=0.1,
            help=texts["temperature_help"],
            key="runtime_temperature_slider",
        )

    if not st.session_state.get("api_key", "").strip() or not st.session_state.get("selected_model"):
        return None

    try:
        return build_openrouter_client(st.session_state["api_key"])
    except Exception:
        return None


def render_chat_history() -> None:
    for msg in st.session_state.get("chat_history", []):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            # =========================================================
# Initial Setup Flow Screens
# =========================================================

def render_ui_language_step() -> None:
    texts = get_ui_texts()

    st.subheader(texts["choose_ui_language"])

    options = list(UI_LANGUAGE_OPTIONS.keys())
    current_value = st.session_state.get("ui_language", "ar")
    if current_value not in options:
        current_value = "ar"

    selected_ui_language = st.selectbox(
        texts["select_language"],
        options=options,
        index=options.index(current_value),
        format_func=lambda x: label_for_ui_language(x),
        key="initial_ui_language_select",
    )

    if st.button(texts["confirm"], key="confirm_ui_language_button"):
        st.session_state.ui_language = selected_ui_language
        st.session_state.state = "awaiting_teaching_language"
        st.rerun()


def render_teaching_language_step() -> None:
    texts = get_ui_texts()

    st.subheader(texts["choose_teaching_language"])

    options = list(TEACHING_LANGUAGE_OPTIONS.keys())
    current_value = st.session_state.get("teaching_language", "ar")
    if current_value not in options:
        current_value = "ar"

    selected_teaching_language = st.selectbox(
        texts["select_language"],
        options=options,
        index=options.index(current_value),
        format_func=lambda x: label_for_teaching_language(x),
        key="initial_teaching_language_select",
    )

    if st.button(texts["confirm"], key="confirm_teaching_language_button"):
        st.session_state.teaching_language = selected_teaching_language
        st.session_state.state = "awaiting_guidance_focus"
        st.rerun()


def render_guidance_focus_step() -> None:
    texts = get_ui_texts()
    ui_lang = st.session_state.get("ui_language", "ar")

    st.subheader(texts["choose_focus"])
    st.caption(texts["focus_hint"])

    focus_options = list(GUIDANCE_FOCUS_OPTIONS[ui_lang].keys())
    current_value = st.session_state.get("guidance_focus", "iso_work_guidance")
    if current_value not in focus_options:
        current_value = "iso_work_guidance"

    selected_focus = st.selectbox(
        texts["select_focus"],
        options=focus_options,
        index=focus_options.index(current_value),
        format_func=lambda x: label_for_focus(x, ui_lang),
        key="initial_guidance_focus_select",
    )

    if st.button(texts["confirm"], key="confirm_guidance_focus_button"):
        st.session_state.guidance_focus = selected_focus
        st.session_state.state = "awaiting_depth"
        st.rerun()


def render_depth_step() -> None:
    texts = get_ui_texts()
    ui_lang = st.session_state.get("ui_language", "ar")

    st.subheader(texts["choose_depth"])

    depth_options = list(LEARNING_DEPTH_OPTIONS[ui_lang].keys())
    current_value = st.session_state.get("depth", "guided")
    if current_value not in depth_options:
        current_value = "guided"

    selected_depth = st.selectbox(
        texts["select_depth"],
        options=depth_options,
        index=depth_options.index(current_value),
        format_func=lambda x: label_for_depth(x, ui_lang),
        key="initial_depth_select",
    )

    if st.button(texts["confirm"], key="confirm_depth_button"):
        st.session_state.depth = selected_depth
        st.session_state.state = "awaiting_material"
        st.rerun()


# =========================================================
# Material Input Helpers
# =========================================================

def read_uploaded_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    try:
        raw_bytes = uploaded_file.read()
    except Exception:
        return ""

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1256", "latin-1"]

    for encoding in encodings_to_try:
        try:
            return raw_bytes.decode(encoding)
        except Exception:
            continue

    return ""


def render_material_step() -> None:
    texts = get_ui_texts()

    st.markdown(f"**{texts['send_material']}**")
    st.subheader(texts["add_material"])

    input_method = st.radio(
        texts["input_method"],
        options=[texts["pasted_text"], texts["upload_file"]],
        key="material_input_method_radio",
    )

    content = ""
    uploaded_name: Optional[str] = None

    if input_method == texts["pasted_text"]:
        content = st.text_area(
            texts["paste_here"],
            height=260,
            key="material_text_area",
        )
    else:
        uploaded_file = st.file_uploader(
            texts["upload_prompt"],
            type=["txt", "md"],
            key="material_file_uploader",
        )

        if uploaded_file is not None:
            uploaded_name = uploaded_file.name
            content = read_uploaded_text_file(uploaded_file)

            if uploaded_file and not content.strip():
                st.error(texts["file_error"])

    if st.button(texts["submit_material"], key="submit_material_button"):
        cleaned = clean_material_text(content)

        if not cleaned.strip():
            st.warning(texts["material_empty"])
            return

        subject = build_new_subject(
            raw_content=content,
            teaching_language=st.session_state.get("teaching_language", "ar"),
            guidance_focus=st.session_state.get("guidance_focus", "iso_work_guidance"),
            depth=st.session_state.get("depth", "guided"),
            input_method="pasted_text" if input_method == texts["pasted_text"] else "upload_file",
            uploaded_name=uploaded_name,
            subjects=st.session_state.get("subjects", {}),
            ui_lang=st.session_state.get("ui_language", "ar"),
        )

        st.session_state.subjects[subject["id"]] = subject
        save_subject(subject)
        load_subject_into_session(subject)
        st.session_state.state = "awaiting_pages"
        st.rerun()


# =========================================================
# Pages Count Step
# =========================================================

def render_pages_step() -> None:
    texts = get_ui_texts()
    subject = st.session_state.get("current_subject")

    if not subject:
        st.error(texts["no_subject_loaded"])
        return

    render_subject_snapshot(subject)
    st.markdown(f"**{texts['pages_question']}**")

    current_total = subject.get("total_pages")
    default_value = int(current_total) if current_total else 3
    if default_value < 1:
        default_value = 1
    if default_value > MAX_PAGES:
        default_value = MAX_PAGES

    pages = st.number_input(
        texts["pages_label"],
        min_value=1,
        max_value=MAX_PAGES,
        value=default_value,
        step=1,
        key="pages_number_input",
    )

    if st.button(texts["confirm"], key="confirm_pages_button"):
        pages_value = int(pages)

        if pages_value > MAX_PAGES:
            st.warning(texts["max_pages_error"])
            return

        subject["total_pages"] = pages_value
        subject["current_page"] = 1
        update_subject_progress(subject)
        save_subject(subject)

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        st.session_state.state = "teaching"
        st.rerun()
        # =========================================================
# Teaching View / Page Navigation / Main Teaching Actions
# =========================================================

def render_current_page_view(subject: Dict[str, Any]) -> None:
    texts = get_ui_texts()

    current_page = int(subject.get("current_page") or 1)
    total_pages = int(subject.get("total_pages") or 1)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown(f"### {texts['page']} {current_page}/{total_pages}")

    if page_exists(subject, current_page):
        st.markdown(get_generated_page_content(subject, current_page))
    else:
        st.info(texts["waiting_content"])

    st.markdown("</div>", unsafe_allow_html=True)


def render_generated_pages_browser(subject: Dict[str, Any]) -> None:
    ui_lang = st.session_state.get("ui_language", "ar")

    generated_pages = [
        item for item in (subject.get("generated_pages", []) or [])
        if (item.get("content") or "").strip()
    ]

    if not generated_pages:
        return

    browser_title = "الصفحات المتولدة" if ui_lang == "ar" else "Generated Pages"

    with st.expander(browser_title, expanded=False):
        total = len(generated_pages)
        per_row = 6

        for start in range(0, total, per_row):
            row_items = generated_pages[start:start + per_row]
            cols = st.columns(len(row_items))

            for col, item in zip(cols, row_items):
                page_index = int(item.get("page_index") or 1)
                with col:
                    if st.button(
                        f"{page_index}",
                        key=f"generated_page_jump_{page_index}",
                    ):
                        subject["current_page"] = page_index
                        save_subject(subject)
                        st.session_state.current_subject = subject
                        st.session_state.subjects[subject["id"]] = subject
                        st.rerun()


def execute_teaching_action(
    result: Dict[str, Any],
    subject: Dict[str, Any],
    client: OpenAI,
    model: str,
    temperature: float,
) -> None:
    texts = get_ui_texts()

    action = result.get("action", "message_only")

    if action == "generate_page":
        requested_page = int(result.get("page_index", subject.get("current_page", 1)))
        subject["current_page"] = requested_page
        save_subject(subject)

        if page_exists(subject, requested_page):
            st.session_state.current_subject = subject
            st.session_state.subjects[subject["id"]] = subject
            st.rerun()

        with st.chat_message("assistant"):
            generate_page_and_update(
                subject=subject,
                page_index=requested_page,
                client=client,
                model=model,
                temperature=temperature,
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        return

    if action == "summary":
        with st.chat_message("assistant"):
            generate_summary_response(
                subject=subject,
                client=client,
                model=model,
                temperature=temperature,
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        return

    if action == "review":
        with st.chat_message("assistant"):
            generate_review_response(
                subject=subject,
                client=client,
                model=model,
                temperature=temperature,
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        return

    if action == "simpler":
        with st.chat_message("assistant"):
            regenerate_current_page_simpler(
                subject=subject,
                client=client,
                model=model,
                temperature=temperature,
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        return

    if action == "deeper":
        with st.chat_message("assistant"):
            regenerate_current_page_deeper(
                subject=subject,
                client=client,
                model=model,
                temperature=temperature,
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        return

    if action == "free_chat":
        user_message = result.get("message", "")
        with st.chat_message("assistant"):
            generate_free_chat_response(
                subject=subject,
                user_input=user_message,
                client=client,
                model=model,
                temperature=temperature,
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        return

    assistant_message = result.get("message", texts["use_commands"])
    with st.chat_message("assistant"):
        st.markdown(assistant_message)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_message,
    })
    sync_subject_chat(subject)
    st.session_state.current_subject = subject
    st.session_state.subjects[subject["id"]] = subject


def handle_next_button_click(
    subject: Dict[str, Any],
    client: Optional[OpenAI],
) -> None:
    texts = get_ui_texts()

    if client is None:
        st.warning(texts["api_warning"])
        return

    total_pages = int(subject.get("total_pages") or 1)
    current_page = int(subject.get("current_page") or 1)

    if current_page >= total_pages:
        return

    next_page = current_page + 1
    subject["current_page"] = next_page
    save_subject(subject)

    if page_exists(subject, next_page):
        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject
        st.rerun()

    with st.chat_message("assistant"):
        generate_page_and_update(
            subject=subject,
            page_index=next_page,
            client=client,
            model=st.session_state.get("selected_model"),
            temperature=float(st.session_state.get("temperature", 0.3)),
        )

    st.session_state.current_subject = subject
    st.session_state.subjects[subject["id"]] = subject


def render_teaching_controls(subject: Dict[str, Any], client: Optional[OpenAI]) -> None:
    texts = get_ui_texts()

    total_pages = int(subject.get("total_pages") or 1)
    current_page = int(subject.get("current_page") or 1)
    next_disabled = current_page >= total_pages

    col_info, col_next = st.columns([3, 1])

    with col_info:
        st.caption(texts["use_commands"])

    with col_next:
        if st.button(
            texts["next_button"],
            key="next_page_main_button",
            disabled=next_disabled,
        ):
            handle_next_button_click(subject, client)


def render_teaching_area() -> None:
    texts = get_ui_texts()
    subject = st.session_state.get("current_subject")

    if not subject:
        st.error(texts["no_subject_loaded"])
        return

    render_subject_snapshot(subject)

    client = render_runtime_settings(subject)

    render_generated_pages_browser(subject)
    render_current_page_view(subject)
    render_teaching_controls(subject, client)

    if client is None:
        st.warning(texts["api_warning"])
        render_chat_history()
        return

    if not subject.get("generated_pages") and subject.get("total_pages"):
        st.info(texts["generate_first_page"])

        with st.chat_message("assistant"):
            generate_page_and_update(
                subject=subject,
                page_index=1,
                client=client,
                model=st.session_state.get("selected_model"),
                temperature=float(st.session_state.get("temperature", 0.3)),
            )

        st.session_state.current_subject = subject
        st.session_state.subjects[subject["id"]] = subject

    render_chat_history()

    prompt = st.chat_input(texts["your_message"])
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        result = process_user_input(
            user_input=prompt,
            state=st.session_state.get("state", "teaching"),
            subject=subject,
            texts=texts,
        )

        st.session_state.state = result.get("next_state", st.session_state.get("state", "teaching"))

        execute_teaching_action(
            result=result,
            subject=subject,
            client=client,
            model=st.session_state.get("selected_model"),
            temperature=float(st.session_state.get("temperature", 0.3)),
        )

    st.markdown("---")
    render_export_actions(subject)
    # =========================================================
# App State Validation / Main Area Routing
# =========================================================

def ensure_valid_runtime_state() -> None:
    current_state = st.session_state.get("state", "awaiting_ui_language")
    current_subject = st.session_state.get("current_subject")

    if current_state not in STATES:
        st.session_state.state = "awaiting_ui_language"
        return

    if current_state in {"awaiting_pages", "teaching", "reviewing", "summarizing"} and not current_subject:
        st.session_state.state = "awaiting_ui_language"
        st.session_state.chat_history = []


def sync_subjects_store_after_runtime_changes() -> None:
    current_subject = st.session_state.get("current_subject")
    if not current_subject:
        return

    subject_id = current_subject.get("id")
    if not subject_id:
        return

    st.session_state.subjects[subject_id] = current_subject


def render_main_area() -> None:
    ensure_valid_runtime_state()

    state = st.session_state.get("state", "awaiting_ui_language")

    if state == "awaiting_ui_language":
        render_ui_language_step()
        return

    if state == "awaiting_teaching_language":
        render_teaching_language_step()
        return

    if state == "awaiting_guidance_focus":
        render_guidance_focus_step()
        return

    if state == "awaiting_depth":
        render_depth_step()
        return

    if state == "awaiting_material":
        render_material_step()
        return

    if state == "awaiting_pages":
        render_pages_step()
        return

    if state in {"teaching", "reviewing", "summarizing"}:
        render_teaching_area()
        sync_subjects_store_after_runtime_changes()
        return

    st.session_state.state = "awaiting_ui_language"
    st.rerun()


# =========================================================
# App Shell Helpers
# =========================================================

def render_app_shell() -> None:
    render_header()
    render_left_panel()
    render_main_area()


def load_runtime_data() -> None:
    st.session_state.subjects = load_subjects()
    refresh_current_subject_reference()

    if not st.session_state.get("iso_knowledge"):
        load_knowledge_into_session()

    # Initialize RAG system only if API key is available
    if not st.session_state.get("rag_initialized") and st.session_state.get("api_key"):
        init_rag_system()


def prepare_app_session() -> None:
    ensure_storage_dirs()
    init_session_state()
    load_runtime_data()
    apply_app_styles()
    ensure_valid_runtime_state()
    # =========================================================
# Final Bootstrap
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    prepare_app_session()
    render_app_shell()


if __name__ == "__main__":
    main()


