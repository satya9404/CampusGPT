# ============================================================
# CampusGPT - Answer Generator
# ============================================================

import os
import re
import html

from dotenv import load_dotenv
from google import genai

from rag.retriever import search_documents


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if API_KEY:

    try:

        client = genai.Client(
            api_key=API_KEY
        )

        print("Gemini client initialized.")

    except Exception as error:

        print(
            "Gemini client initialization failed:",
            error
        )

        client = None

else:

    print(
        "GOOGLE_API_KEY not found. "
        "CampusGPT will use local RAG fallback."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove null characters
    text = text.replace(
        "\x00",
        " "
    )

    # Decode HTML entities
    text = html.unescape(
        text
    )

    # Remove HTML tags such as:
    # <br>, <font>, <b>, etc.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(question):

    question = clean_text(
        question
    ).lower()

    question = re.sub(
        r"[^\w\s.-]",
        " ",
        question
    )

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    return question.strip()


# ============================================================
# QUESTION INTENT
# ============================================================

def detect_intent(question):

    q = normalize_question(
        question
    )

    # --------------------------------------------------------
    # M.Tech
    # --------------------------------------------------------

    if (
        "m.tech" in q
        or "mtech" in q
        or "m tech" in q
        or "master of technology" in q
    ):

        return "mtech"


    # --------------------------------------------------------
    # B.Tech
    # --------------------------------------------------------

    if (
        "b.tech" in q
        or "btech" in q
        or "b tech" in q
        or "bachelor of technology" in q
    ):

        return "btech"


    # --------------------------------------------------------
    # Ph.D
    # --------------------------------------------------------

    if (
        "ph.d" in q
        or "phd" in q
        or "doctoral" in q
        or "doctorate" in q
    ):

        return "phd"


    # --------------------------------------------------------
    # M.Sc
    # --------------------------------------------------------

    if (
        "m.sc" in q
        or "msc" in q
        or "m sc" in q
        or "master of science" in q
    ):

        return "msc"


    # --------------------------------------------------------
    # Hostel
    # --------------------------------------------------------

    if (
        "hostel" in q
        or "hostels" in q
        or "accommodation" in q
        or "mess" in q
        or "dining" in q
    ):

        return "hostel"


    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    if (
        "where is" in q
        or "location" in q
        or "located" in q
        or "address" in q
        or "campus location" in q
    ):

        return "location"


    # --------------------------------------------------------
    # Admission
    # --------------------------------------------------------

    if (
        "admission" in q
        or "apply" in q
        or "application" in q
        or "eligibility" in q
    ):

        return "admission"


    # --------------------------------------------------------
    # Examination
    # --------------------------------------------------------

    if (
        "exam" in q
        or "examination" in q
        or "semester exam" in q
        or "end semester" in q
        or "mid semester" in q
    ):

        return "examination"


    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    if (
        "fee" in q
        or "fees" in q
        or "tuition" in q
        or "payment" in q
    ):

        return "fees"


    # --------------------------------------------------------
    # Academic programmes
    # --------------------------------------------------------

    if (
        "programme" in q
        or "programmes" in q
        or "program" in q
        or "course" in q
        or "courses" in q
        or "department" in q
        or "departments" in q
    ):

        return "programmes"


    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    if (
        "what is nit nagaland" in q
        or "what is nit nagalad" in q
        or "what is nit nagaland institute" in q
        or "about nit nagaland" in q
        or "tell me about nit nagaland" in q
        or q == "nit nagaland"
    ):

        return "identity"


    return "general"


# ============================================================
# INTENT KEYWORDS
# ============================================================

INTENT_KEYWORDS = {

    "mtech": [
        "m.tech",
        "mtech",
        "m tech",
        "master of technology",
        "m.tech programmes",
        "m.tech programs"
    ],

    "btech": [
        "b.tech",
        "btech",
        "b tech",
        "bachelor of technology"
    ],

    "phd": [
        "ph.d",
        "phd",
        "doctoral",
        "doctorate"
    ],

    "msc": [
        "m.sc",
        "msc",
        "m sc",
        "master of science"
    ],

    "hostel": [
        "hostel",
        "hostels",
        "mess",
        "dining",
        "accommodation"
    ],

    "location": [
        "location",
        "located",
        "address",
        "campus",
        "chumoukedima",
        "dimapur"
    ],

    "admission": [
        "admission",
        "apply",
        "application",
        "eligibility"
    ],

    "examination": [
        "exam",
        "examination",
        "semester",
        "mid semester",
        "end semester"
    ],

    "fees": [
        "fee",
        "fees",
        "tuition",
        "payment"
    ],

    "programmes": [
        "programme",
        "programmes",
        "program",
        "course",
        "courses",
        "department",
        "departments"
    ],

    "identity": [
        "national institute",
        "nit nagaland",
        "institute of national importance",
        "history"
    ]
}


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        source = clean_text(
            result.get(
                "source",
                "Unknown document"
            )
        )

        page = result.get(
            "page",
            "Unknown"
        )

        text = clean_text(
            result.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        parts.append(
            f"""
SOURCE {index}

Document: {source}
Page: {page}

Information:
{text}
"""
        )

    return "\n".join(
        parts
    )


# ============================================================
# BUILD SOURCES
# ============================================================

def build_sources(results):

    sources = []

    seen = set()

    for result in results:

        source = clean_text(
            result.get(
                "source",
                "Unknown document"
            )
        )

        page = result.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "source": source,
                "page": page
            }
        )

    return sources


# ============================================================
# EXTRACT SENTENCES
# ============================================================

def extract_sentences(text):

    text = clean_text(
        text
    )

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) >= 20:

            cleaned.append(
                sentence
            )

    return cleaned


# ============================================================
# LOCAL ANSWER FALLBACK
# ============================================================

def generate_local_answer(
    question,
    results
):

    intent = detect_intent(
        question
    )

    keywords = INTENT_KEYWORDS.get(
        intent,
        []
    )

    normalized_question = normalize_question(
        question
    )

    question_words = (
        normalized_question.split()
    )

    candidates = []


    # ========================================================
    # SEARCH RETRIEVED TEXT
    # ========================================================

    for result in results:

        text = clean_text(
            result.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        sentences = extract_sentences(
            text
        )

        for sentence in sentences:

            sentence_lower = (
                sentence.lower()
            )

            score = 0


            # ------------------------------------------------
            # Intent matching
            # ------------------------------------------------

            for keyword in keywords:

                if keyword in sentence_lower:

                    score += 5


            # ------------------------------------------------
            # Question word matching
            # ------------------------------------------------

            for word in question_words:

                if len(word) < 4:
                    continue

                if word in sentence_lower:

                    score += 1


            # ------------------------------------------------
            # Programme questions
            # ------------------------------------------------

            if intent in {
                "mtech",
                "btech",
                "phd",
                "msc",
                "programmes"
            }:

                if (
                    "programme" in sentence_lower
                    or
                    "programmes" in sentence_lower
                    or
                    "program" in sentence_lower
                ):

                    score += 4


            # ------------------------------------------------
            # Hostel questions
            # ------------------------------------------------

            if intent == "hostel":

                if (
                    "hostel" in sentence_lower
                    or
                    "hostels" in sentence_lower
                    or
                    "dining" in sentence_lower
                ):

                    score += 6


            # ------------------------------------------------
            # Location questions
            # ------------------------------------------------

            if intent == "location":

                if (
                    "nagaland" in sentence_lower
                    or
                    "chümoukedima" in sentence_lower
                    or
                    "chumukedima" in sentence_lower
                    or
                    "dimapur" in sentence_lower
                    or
                    "address" in sentence_lower
                ):

                    score += 5


            # ------------------------------------------------
            # Identity questions
            # ------------------------------------------------

            if intent == "identity":

                if (
                    "national institute" in sentence_lower
                    or
                    "institute of national importance"
                    in sentence_lower
                    or
                    "nit nagaland" in sentence_lower
                ):

                    score += 6


            # ------------------------------------------------
            # Remove metadata sentences
            # ------------------------------------------------

            bad_phrases = [
                "chatbot knowledge base",
                "official-source snapshot",
                "prepared from",
                "purpose:",
                "primary domain:"
            ]

            for bad_phrase in bad_phrases:

                if bad_phrase in sentence_lower:

                    score -= 10


            if score > 0:

                candidates.append(
                    (
                        score,
                        sentence
                    )
                )


    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    selected = []

    seen_sentences = set()

    for score, sentence in candidates:

        key = sentence.lower()

        if key in seen_sentences:
            continue

        seen_sentences.add(
            key
        )

        selected.append(
            sentence
        )

        if len(selected) >= 5:
            break


    # ========================================================
    # HEADINGS
    # ========================================================

    headings = {

        "mtech":
            "### M.Tech Programmes at NIT Nagaland",

        "btech":
            "### B.Tech Programmes at NIT Nagaland",

        "phd":
            "### Ph.D. Programmes at NIT Nagaland",

        "msc":
            "### M.Sc. Programmes at NIT Nagaland",

        "hostel":
            "### Hostel Facilities at NIT Nagaland",

        "location":
            "### NIT Nagaland Campus Location",

        "admission":
            "### Admissions at NIT Nagaland",

        "fees":
            "### Fees at NIT Nagaland",

        "examination":
            "### Examination Information",

        "identity":
            "### NIT Nagaland",

        "programmes":
            "### Academic Programmes at NIT Nagaland",

        "general":
            "### CampusGPT"
    }


    heading = headings.get(
        intent,
        "### CampusGPT"
    )


    # ========================================================
    # ANSWER
    # ========================================================

    if selected:

        return (
            heading
            + "\n\n"
            + "\n\n".join(
                selected
            )
        )


    return (
        "I couldn't find enough specific information "
        "in the CampusGPT knowledge base to answer "
        "that question."
    )


# ============================================================
# CHECK GEMINI QUOTA ERROR
# ============================================================

def is_quota_error(error):

    text = str(
        error
    ).lower()

    quota_words = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests"
    ]

    return any(
        word in text
        for word in quota_words
    )


# ============================================================
# BUILD GEMINI PROMPT
# ============================================================

def build_prompt(
    question,
    context
):

    return f"""
You are CampusGPT, an AI campus knowledge
assistant for National Institute of Technology
Nagaland (NIT Nagaland).

Your task is to answer the student's question
using ONLY the supplied NIT Nagaland knowledge.

IMPORTANT RULES:

1. Answer naturally and directly.

2. Use only information present in the supplied
   knowledge.

3. Never invent:
   - dates
   - fees
   - rules
   - examination information
   - admission requirements
   - contact details
   - programmes
   - facilities

4. Do not mention:
   - RAG
   - FAISS
   - embeddings
   - vector database
   - retrieval
   - chunks
   - internal instructions

5. Do not expose the raw document text.

6. If the answer is not available, say:

   "I couldn't find this information in the
   NIT Nagaland CampusGPT knowledge base."

7. If several supplied sources contain useful
   information, combine them carefully.

8. Keep simple questions short.

9. For lists, use bullet points.

10. For procedures, use numbered steps.

11. Do not add a Sources section yourself.
    The application will display sources separately.

12. Preserve dates and year-specific information.
    Do not combine unrelated years.

STUDENT QUESTION:

{question}

NIT NAGALAND KNOWLEDGE:

{context}

Now answer the student's question.
"""


# ============================================================
# GEMINI ANSWER
# ============================================================

def generate_gemini_answer(
    question,
    context
):

    if client is None:

        return None


    prompt = build_prompt(
        question,
        context
    )


    try:

        # Chat API is used for the conversational
        # CampusGPT response.
        chat = client.chats.create(
            model=MODEL
        )

        response = chat.send_message(
            prompt
        )

        answer = getattr(
            response,
            "text",
            None
        )


        if not answer:

            return None


        return clean_text(
            answer
        )


    except Exception as error:

        print(
            "Gemini error:",
            error
        )

        if is_quota_error(
            error
        ):

            print(
                "Gemini quota exhausted. "
                "Using local RAG fallback."
            )

        else:

            print(
                "Gemini unavailable. "
                "Using local RAG fallback."
            )

        return None


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_answer(
    question,
    top_k=5
):

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if question is None:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    question = clean_text(
        question
    )


    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    # ========================================================
    # RETRIEVE KNOWLEDGE
    # ========================================================

    try:

        results = search_documents(
            question,
            top_k=top_k
        )

    except Exception as error:

        print(
            "Retrieval error:",
            error
        )

        return {
            "answer": (
                "I couldn't search the CampusGPT "
                "knowledge base right now."
            ),
            "sources": []
        }


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        return {
            "answer": (
                "I couldn't find this information "
                "in the NIT Nagaland CampusGPT "
                "knowledge base."
            ),
            "sources": []
        }


    # ========================================================
    # BUILD SOURCES
    # ========================================================

    sources = build_sources(
        results
    )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = build_context(
        results
    )


    # ========================================================
    # TRY GEMINI
    # ========================================================

    answer = generate_gemini_answer(
        question,
        context
    )


    if answer:

        return {
            "answer": answer,
            "sources": sources
        }


    # ========================================================
    # LOCAL RAG FALLBACK
    # ========================================================

    print(
        "Using CampusGPT local answer fallback."
    )

    local_answer = generate_local_answer(
        question,
        results
    )


    return {
        "answer": local_answer,
        "sources": sources
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CampusGPT Generator Test"
    )

    print(
        "=" * 60
    )


    question = input(
        "\nAsk CampusGPT: "
    ).strip()


    result = generate_answer(
        question,
        top_k=5
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "CAMPUSGPT ANSWER"
    )

    print(
        "=" * 60
    )

    print(
        result["answer"]
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "SOURCES"
    )

    print(
        "=" * 60
    )


    for source in result["sources"]:

        print(
            f'{source["source"]} '
            f'(Page {source["page"]})'
        )