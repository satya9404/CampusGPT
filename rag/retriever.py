# ============================================================
# CampusGPT - Smart Intent-Aware Document Retriever
# ============================================================

import re
import numpy as np

from rag.embeddings import create_embedding
from rag.vector_store import load_vector_store


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_TOP_K = 5
CANDIDATE_COUNT = 30
MAX_DISTANCE = 1.20


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "to",
    "for",
    "on",
    "and",
    "or",
    "how",
    "who",
    "when",
    "where",
    "why",
    "which",
    "can",
    "does",
    "do",
    "about",
    "tell",
    "me",
    "please",
    "tell",
}


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into useful lowercase words.
    """

    if not text:
        return []

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    return " ".join(
        text.lower().split()
    )


# ============================================================
# DETECT QUESTION INTENT
# ============================================================

def detect_intent(query):

    query_lower = query.lower().strip()


    # --------------------------------------------------------
    # GENERAL NIT NAGALAND
    # --------------------------------------------------------

    general_phrases = [
        "what is nit nagaland",
        "what is nit nagalad",
        "what is nit",
        "about nit nagaland",
        "about nit nagalad",
        "tell me about nit nagaland",
        "tell me about nit",
        "nit nagaland",
        "nit nagalad",
        "about the institute",
        "about the college",
        "institute history",
        "history of nit nagaland",
        "history of nit",
    ]

    if any(
        phrase in query_lower
        for phrase in general_phrases
    ):
        return "general"


    # --------------------------------------------------------
    # PHD
    # --------------------------------------------------------

    phd_words = [
        "phd",
        "ph.d",
        "doctoral",
        "research scholar",
        "doctoral programme",
        "doctoral program",
    ]

    if any(
        word in query_lower
        for word in phd_words
    ):
        return "phd"


    # --------------------------------------------------------
    # MTECH / MSC
    # --------------------------------------------------------

    postgraduate_words = [
        "mtech",
        "m.tech",
        "m sc",
        "m.sc",
        "msc",
        "postgraduate",
        "post graduate",
    ]

    if any(
        word in query_lower
        for word in postgraduate_words
    ):
        return "postgraduate"


    # --------------------------------------------------------
    # ADMISSION
    # --------------------------------------------------------

    admission_words = [
        "admission",
        "eligibility",
        "application",
        "apply",
        "applying",
        "entrance",
        "selection",
        "candidate",
    ]

    if any(
        word in query_lower
        for word in admission_words
    ):
        return "admission"


    # --------------------------------------------------------
    # ACADEMIC
    # --------------------------------------------------------

    academic_words = [
        "academic calendar",
        "semester",
        "course",
        "courses",
        "syllabus",
        "department",
        "programme",
        "program",
        "btech",
        "b.tech",
        "degree",
        "curriculum",
    ]

    if any(
        word in query_lower
        for word in academic_words
    ):
        return "academic"


    # --------------------------------------------------------
    # HOSTEL
    # --------------------------------------------------------

    hostel_words = [
        "hostel",
        "hostels",
        "mess",
        "dining",
        "accommodation",
        "room",
    ]

    if any(
        word in query_lower
        for word in hostel_words
    ):
        return "hostel"


    # --------------------------------------------------------
    # PLACEMENT
    # --------------------------------------------------------

    placement_words = [
        "placement",
        "placements",
        "recruitment",
        "company",
        "companies",
        "package",
        "salary",
        "career",
    ]

    if any(
        word in query_lower
        for word in placement_words
    ):
        return "placement"


    # --------------------------------------------------------
    # FACULTY
    # --------------------------------------------------------

    faculty_words = [
        "faculty",
        "professor",
        "professors",
        "teacher",
        "teachers",
        "hod",
        "head of department",
    ]

    if any(
        word in query_lower
        for word in faculty_words
    ):
        return "faculty"


    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    contact_words = [
        "contact",
        "phone",
        "email",
        "address",
        "location",
        "website",
    ]

    if any(
        word in query_lower
        for word in contact_words
    ):
        return "contact"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "general"


# ============================================================
# INTENT SCORE
# ============================================================

def intent_score(query, result):

    intent = detect_intent(
        query
    )

    source = result[
        "source"
    ].lower()

    text = result[
        "text"
    ].lower()


    score = 0.0


    # ========================================================
    # GENERAL
    # ========================================================

    if intent == "general":

        # Dedicated NIT knowledge file gets
        # a strong preference.

        if (
            "nit_nagaland_template" in source
            or
            "nit nagaland template" in source
        ):
            score += 0.40


        general_terms = [
            "national institute of technology nagaland",
            "nit nagaland",
            "institute of national importance",
            "institute identity",
            "institute history",
        ]

        for term in general_terms:

            if term in text:
                score += 0.08


    # ========================================================
    # PHD
    # ========================================================

    elif intent == "phd":

        if "phd" in source:
            score += 0.30

        if "ph.d" in source:
            score += 0.30

        if "doctoral" in source:
            score += 0.20

        if "phd" in text:
            score += 0.10


    # ========================================================
    # POSTGRADUATE
    # ========================================================

    elif intent == "postgraduate":

        if "mtech" in source:
            score += 0.30

        if "m.tech" in source:
            score += 0.30

        if "msc" in source:
            score += 0.20

        if "m.sc" in source:
            score += 0.20


        if "m.tech" in text:
            score += 0.10

        if "mtech" in text:
            score += 0.10


    # ========================================================
    # ADMISSION
    # ========================================================

    elif intent == "admission":

        admission_terms = [
            "admission",
            "eligibility",
            "application",
            "notification",
            "candidate",
        ]

        for term in admission_terms:

            if term in text:
                score += 0.06


    # ========================================================
    # ACADEMIC
    # ========================================================

    elif intent == "academic":

        academic_terms = [
            "academic",
            "semester",
            "course",
            "syllabus",
            "department",
            "programme",
            "program",
            "b.tech",
            "btech",
        ]

        for term in academic_terms:

            if term in text:
                score += 0.05


    # ========================================================
    # HOSTEL
    # ========================================================

    elif intent == "hostel":

        if "hostel" in text:
            score += 0.25

        if "dining" in text:
            score += 0.10

        if "mess" in text:
            score += 0.10


    # ========================================================
    # PLACEMENT
    # ========================================================

    elif intent == "placement":

        if "placement" in text:
            score += 0.25

        if "recruitment" in text:
            score += 0.15

        if "company" in text:
            score += 0.05


    # ========================================================
    # FACULTY
    # ========================================================

    elif intent == "faculty":

        if "faculty" in text:
            score += 0.25

        if "professor" in text:
            score += 0.15

        if "hod" in text:
            score += 0.10


    # ========================================================
    # CONTACT
    # ========================================================

    elif intent == "contact":

        contact_terms = [
            "contact",
            "email",
            "phone",
            "address",
            "website",
        ]

        for term in contact_terms:

            if term in text:
                score += 0.08


    return score


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    query,
    top_k=DEFAULT_TOP_K
):

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query:
        return []


    query = query.strip()


    if not query:
        return []


    # --------------------------------------------------------
    # Load FAISS vector store
    # --------------------------------------------------------

    try:

        index, chunks = load_vector_store()

    except Exception as error:

        print(
            "Vector store error:",
            error
        )

        return []


    if index is None:
        return []


    if not chunks:
        return []


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    try:

        embedding = create_embedding(
            query
        )

    except Exception as error:

        print(
            "Embedding error:",
            error
        )

        return []


    # --------------------------------------------------------
    # Convert embedding to NumPy
    # --------------------------------------------------------

    query_vector = np.asarray(
        embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # Make shape compatible with FAISS
    # --------------------------------------------------------

    if query_vector.ndim == 1:

        query_vector = query_vector.reshape(
            1,
            -1
        )


    if query_vector.ndim != 2:

        print(
            "Invalid embedding shape:",
            query_vector.shape
        )

        return []


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if query_vector.shape[1] != index.d:

        print(
            "Embedding dimension mismatch."
        )

        print(
            "Query dimension:",
            query_vector.shape[1]
        )

        print(
            "FAISS dimension:",
            index.d
        )

        return []


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    search_count = min(
        CANDIDATE_COUNT,
        len(chunks)
    )


    try:

        distances, indices = index.search(
            query_vector,
            search_count
        )

    except Exception as error:

        print(
            "FAISS search error:",
            error
        )

        return []


    # --------------------------------------------------------
    # Build candidate list
    # --------------------------------------------------------

    candidates = []

    seen = set()


    for distance, chunk_index in zip(
        distances[0],
        indices[0]
    ):

        if chunk_index < 0:
            continue


        if chunk_index >= len(chunks):
            continue


        chunk = chunks[
            int(chunk_index)
        ]


        text = chunk.get(
            "text",
            ""
        )


        source = chunk.get(
            "source",
            "Unknown document"
        )


        page = chunk.get(
            "page",
            "Unknown"
        )


        text = text.strip()


        if not text:
            continue


        distance = float(
            distance
        )


        # ----------------------------------------------------
        # Ignore very weak results
        # ----------------------------------------------------

        if distance > MAX_DISTANCE:
            continue


        # ----------------------------------------------------
        # Duplicate protection
        # ----------------------------------------------------

        key = (
            source,
            page,
            normalize_text(
                text
            )[:500]
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        candidates.append(
            {
                "source": source,
                "page": page,
                "text": text,
                "distance": distance
            }
        )


    # --------------------------------------------------------
    # No useful results
    # --------------------------------------------------------

    if not candidates:
        return []


    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    query_words = set(
        tokenize(query)
    )


    for result in candidates:

        text_words = set(
            tokenize(
                result["text"]
            )
        )


        # ----------------------------------------------------
        # Word overlap
        # ----------------------------------------------------

        overlap = len(
            query_words.intersection(
                text_words
            )
        )


        overlap_score = min(
            overlap * 0.03,
            0.20
        )


        # ----------------------------------------------------
        # Semantic distance score
        # ----------------------------------------------------

        distance_score = max(
            0.0,
            1.0 - result["distance"]
        )


        # ----------------------------------------------------
        # Intent score
        # ----------------------------------------------------

        semantic_intent_score = intent_score(
            query,
            result
        )


        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        result["score"] = (
            distance_score
            + overlap_score
            + semantic_intent_score
        )


    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # ========================================================
    # QUESTION INTENT
    # ========================================================

    intent = detect_intent(
        query
    )


    selected = []


    # ========================================================
    # GENERAL QUESTIONS
    #
    # Example:
    # "What is NIT Nagaland?"
    #
    # Prefer the dedicated NIT knowledge document.
    # ========================================================

    if intent == "general":

        general_results = [
            item
            for item in candidates
            if (
                "nit_nagaland_template"
                in item["source"].lower()
            )
        ]


        general_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )


        selected.extend(
            general_results[:top_k]
        )


    # ========================================================
    # SPECIFIC QUESTIONS
    # ========================================================

    else:

        selected.extend(
            candidates[:top_k]
        )


    # ========================================================
    # FALLBACK
    # ========================================================

    if not selected:

        selected = candidates[
            :top_k
        ]


    # ========================================================
    # FINAL SORT
    # ========================================================

    selected.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    # ========================================================
    # RETURN
    # ========================================================

    return selected[
        :top_k
    ]