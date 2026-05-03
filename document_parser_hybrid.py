"""
OptiScholar — Final Hybrid Document Parser
============================================
Primary  : venkatasagar/NER-roBERTa-finetuned
           Trained on real resume/CV data
           Extracts: NAME, GPA, DEGREE, FIELD, INSTITUTION
Fallback : regex layer
           Fills gaps: GPA from module marks, AGE from DOB,
           GENDER, degree_level category, field from Award Programme

Label mapping inferred from model behaviour (author omitted label names):
  LABEL_38  → B-NAME
  LABEL_93  → I-NAME
  LABEL_27  → B-GPA_KEY  (the word "GPA"/"CGPA")
  LABEL_86  → B-GPA_VALUE
  LABEL_11  → B-GPA_VALUE (alternate)
  LABEL_18  → B-DEGREE
  LABEL_76  → I-DEGREE
  LABEL_25  → B-FIELD
  LABEL_83  → I-FIELD
  LABEL_40  → B-INSTITUTION
  LABEL_94  → I-INSTITUTION
  LABEL_92  → B-LOCATION
  LABEL_111 → O (outside)
"""

import re
import os
import datetime
import warnings

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Model ──────────────────────────────────────────────────────
ROBERTA_MODEL = "fine_tuned_roberta"

# Inferred label mapping
LABEL_MAP = {
    "LABEL_38":  "B-NAME",
    "LABEL_93":  "I-NAME",
    "LABEL_27":  "B-GPA_KEY",
    "LABEL_86":  "B-GPA_VALUE",
    "LABEL_11":  "B-GPA_VALUE",
    "LABEL_18":  "B-DEGREE",
    "LABEL_76":  "I-DEGREE",
    "LABEL_25":  "B-FIELD",
    "LABEL_83":  "I-FIELD",
    "LABEL_40":  "B-INSTITUTION",
    "LABEL_94":  "I-INSTITUTION",
    "LABEL_92":  "B-LOCATION",
    "LABEL_111": "O",
}

# Bad words that indicate a false positive name/field match
BAD_NAMES  = ['transcript', 'university', 'interim', 'official',
               'report', 'system', 'information', 'college',
               'institute', 'ibn', 'certificate', 'academic']
BAD_FIELDS = ['summary', 'code', 'status', 'active', 'current',
               'the', 'and', 'with', 'uk', 'faculty']
FALSE_DEGREES = ['faculty', 'school', 'college', 'department']


# ══════════════════════════════════════════════════════════════
# SECTION 1 — PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_text(pdf_path: str) -> tuple:
    """PyMuPDF → pdfplumber → pytesseract OCR."""
    print(f"\n  Extracting: {Path(pdf_path).name}")

    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = "\n".join(p.get_text("text") for p in doc)
        doc.close()
        if len(text.strip()) >= 50:
            print(f"  [PyMuPDF] {len(text):,} chars")
            return text.strip(), "pymupdf"
    except Exception as e:
        print(f"  [PyMuPDF] {e}")

    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
        if len(text.strip()) >= 50:
            print(f"  [pdfplumber] {len(text):,} chars")
            return text.strip(), "pdfplumber"
    except Exception as e:
        print(f"  [pdfplumber] {e}")

    try:
        import pytesseract
        from pdf2image import convert_from_path
        print("  [OCR] Scanned document — applying pytesseract...")
        pages = convert_from_path(pdf_path, dpi=200)
        text  = "\n".join(pytesseract.image_to_string(p) for p in pages)
        if len(text.strip()) >= 50:
            print(f"  [OCR] {len(text):,} chars")
            return text.strip(), "ocr"
    except ImportError:
        print("  [OCR] pip install pytesseract pdf2image")
    except Exception as e:
        print(f"  [OCR] {e}")

    return "", "failed"


# ══════════════════════════════════════════════════════════════
# SECTION 2 — ROBERTA NER EXTRACTOR (PRIMARY)
# ══════════════════════════════════════════════════════════════

_roberta_pipeline = None


def get_roberta_pipeline():
    global _roberta_pipeline
    if _roberta_pipeline is None:
        try:
            from transformers import pipeline
            print(f"  [RoBERTa] Loading {ROBERTA_MODEL}...")
            _roberta_pipeline = pipeline(
                "ner",
                model=ROBERTA_MODEL,
                aggregation_strategy="simple",
                device=0 if __import__('torch').cuda.is_available() else -1
            )
            print("  [RoBERTa] Loaded.")
        except Exception as e:
            print(f"  [RoBERTa] {e}")
    return _roberta_pipeline


def extract_with_roberta(text: str) -> dict:
    """
    Run RoBERTa NER on first page of text (avoids repetition
    from multi-page transcripts) and map to structured fields.
    """
    nlp = get_roberta_pipeline()
    if nlp is None:
        return {}

    # Use only first ~1500 chars to avoid repeated page headers
    # and grade legend sections that cause false positives
    cutoff = min(
        1500,
        text.lower().find("assessment notation") if "assessment notation" in text.lower() else 1500,
        text.lower().find("high distinction")     if "high distinction"    in text.lower() else 1500,
        text.lower().find("\n\n\n")               if "\n\n\n" in text else 1500,
    )
    excerpt = text[:max(cutoff, 300)]

    try:
        raw_entities = nlp(excerpt)
    except Exception as e:
        print(f"  [RoBERTa] Inference error: {e}")
        return {}

    # Group by mapped label
    grouped = {}
    for ent in raw_entities:
        raw_label = ent["entity_group"]
        mapped    = LABEL_MAP.get(raw_label, "O")
        if mapped == "O":
            continue
        score = float(ent["score"])
        word  = ent["word"].strip()
        if len(word) < 2 or score < 0.5:
            continue
        if mapped not in grouped:
            grouped[mapped] = []
        grouped[mapped].append({"text": word, "score": score})

    return grouped


def roberta_to_fields(grouped: dict) -> dict:
    """Convert RoBERTa grouped entities to structured profile fields."""
    fields = {}

    # ── Person name ────────────────────────────────────────────
    name_parts = []
    for label in ["B-NAME", "I-NAME"]:
        if label in grouped:
            for c in grouped[label]:
                if not any(b in c["text"].lower() for b in BAD_NAMES):
                    name_parts.append((c["text"], c["score"]))

    if name_parts:
        name = " ".join(p[0] for p in name_parts)
        conf = sum(p[1] for p in name_parts) / len(name_parts)
        words = name.split()
        if len(words) >= 2 and all(len(w) >= 2 for w in words):
            fields["person_name"] = {
                "value": name, "confidence": round(conf, 3),
                "source": "roberta"
            }

    # ── GPA value ──────────────────────────────────────────────
    # The model tags "GPA" as GPA_KEY and the number as GPA_VALUE
    for label in ["B-GPA_VALUE"]:
        if label in grouped:
            best = max(grouped[label], key=lambda x: x["score"])
            raw  = best["text"].strip(".:,/")
            # Handle "3.45/4.00" → take numerator
            if "/" in raw:
                parts = raw.split("/")
                try:
                    val = float(parts[0])
                    den = float(parts[1])
                    gpa = round((val / den) * 4.0, 2) if den > 4.0 else val
                except ValueError:
                    gpa = None
            else:
                try:
                    gpa = float(raw)
                    # Australian 7.0 scale
                    if gpa > 4.0:
                        gpa = round((gpa / 7.0) * 4.0, 2)
                except ValueError:
                    gpa = None

            if gpa and 0.0 <= gpa <= 4.0:
                fields["final_gpa"] = {
                    "value": round(gpa, 2),
                    "confidence": round(best["score"], 3),
                    "source": "roberta"
                }

    # ── Degree level ───────────────────────────────────────────
    degree_text = ""
    if "B-DEGREE" in grouped:
        # Exclude false positives like "Faculty", "School"
        valid = [c for c in grouped["B-DEGREE"]
                 if not any(f in c["text"].lower()
                            for f in FALSE_DEGREES)]
        if valid:
            best = max(valid, key=lambda x: x["score"])
            degree_parts = [best["text"]]
            if "I-DEGREE" in grouped:
                degree_parts += [c["text"] for c in grouped["I-DEGREE"]]
            degree_text = " ".join(degree_parts)

            tl = degree_text.lower()
            if any(w in tl for w in ["ph.d", "phd", "doctor"]):
                level = "phd"
            elif any(w in tl for w in ["master", "msc", "mba"]):
                level = "master"
            elif any(w in tl for w in ["bachelor", "b.sc", "bsc",
                                         "b.tech", "b.eng"]):
                level = "bachelor"
            else:
                level = None

            if level:
                fields["degree_level"] = {
                    "value": level,
                    "confidence": round(best["score"] * 0.9, 3),
                    "source": "roberta"
                }

    # ── Field of study ─────────────────────────────────────────
    field_parts = []
    for label in ["B-FIELD", "I-FIELD"]:
        if label in grouped:
            for c in grouped[label]:
                field_parts.append((c["text"], c["score"]))

    if field_parts:
        field = " ".join(p[0] for p in field_parts).title()
        conf  = sum(p[1] for p in field_parts) / len(field_parts)
        bad   = any(re.search(r'\b'+w+r'\b', field.lower())
                    for w in BAD_FIELDS)
        if 3 <= len(field) <= 80 and not bad:
            fields["field_of_study"] = {
                "value": field, "confidence": round(conf, 3),
                "source": "roberta"
            }

    # ── Institution ────────────────────────────────────────────
    inst_parts = []
    for label in ["B-INSTITUTION", "I-INSTITUTION"]:
        if label in grouped:
            for c in grouped[label]:
                inst_parts.append((c["text"], c["score"]))

    # Append location if present (e.g. "British University" + "Egypt")
    if "B-LOCATION" in grouped and inst_parts:
        loc = grouped["B-LOCATION"][0]["text"]
        if loc.lower() not in ["uk", "usa", "us"]:
            inst_parts.append((f"in {loc}",
                                grouped["B-LOCATION"][0]["score"]))

    if inst_parts:
        inst = " ".join(p[0] for p in inst_parts)
        conf = sum(p[1] for p in inst_parts) / len(inst_parts)
        fields["institution"] = {
            "value": inst, "confidence": round(conf, 3),
            "source": "roberta"
        }

    return fields


# ══════════════════════════════════════════════════════════════
# SECTION 3 — REGEX FALLBACK
# ══════════════════════════════════════════════════════════════

def regex_gpa(text: str) -> tuple:
    tl = text.lower()

    # Australian "Current GPA: x.xx"
    m = re.search(r'current\s+gpa[\s:]+(\d+\.?\d{1,2})', tl)
    if m:
        val = float(m.group(1))
        gpa = round((val / 7.0) * 4.0, 2) if val > 4.0 else round(val, 2)
        return gpa, 0.88

    patterns = [
        (r'c?gpa[\s:]+(\d+\.?\d*)\s*/\s*(4\.0+|5\.0+|7\.0+)', 'frac'),
        (r'\bc?gpa[\s:=]+(\d+\.?\d{1,2})\b',                   'direct'),
        (r'cumulative\s+(?:gpa|grade\s+point)[\s:]+(\d+\.?\d{1,2})', 'direct'),
        (r'grade\s+point\s+average[\s:]+(\d+\.?\d{1,2})',       'direct'),
        (r'(\d+\.?\d{1,2})\s*/\s*4\.0+',                        'frac4'),
        (r'(\d+\.?\d{1,2})\s*/\s*7\.0+',                        'frac7'),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, tl)
        if m:
            try:
                if fmt == 'frac':
                    gpa = (float(m.group(1)) / float(m.group(2))) * 4.0
                elif fmt == 'frac4':
                    gpa = float(m.group(1))
                elif fmt == 'frac7':
                    gpa = (float(m.group(1)) / 7.0) * 4.0
                else:
                    gpa = float(m.group(1))
                if 0.0 <= gpa <= 4.0: return round(gpa, 2), 0.90
                if 4.0 < gpa <= 7.0:  return round((gpa/7.0)*4.0, 2), 0.80
            except (ValueError, IndexError):
                continue

    # UK/BUE: compute from module marks
    if any(kw in tl for kw in ["degree year","module","marking","award programme"]):
        marks = [float(x) for x in re.findall(r'(?<!\d)([4-9]\d|100)(?!\d)', tl)
                 if 40 <= float(x) <= 100]
        if len(marks) >= 3:
            avg = sum(marks) / len(marks)
            gpa = (4.0 if avg>=70 else 3.3 if avg>=60
                   else 2.7 if avg>=50 else 2.0)
            return round(gpa, 2), 0.82

    return None, 0.0


def regex_degree_level(text: str) -> tuple:
    tl = text.lower()

    # Cut at grade legend
    cut = min(
        tl.find("assessment notation") if "assessment notation" in tl else len(tl),
        tl.find("high distinction")    if "high distinction"    in tl else len(tl),
    )
    clean = tl[:cut] if cut > 100 else tl

    if re.search(r'degree\s+year\s+(one|two|three|four|five|\d+)', clean):
        return "bachelor", 0.92

    m = re.search(r'program\s+name\s+([a-z\s]+?)(?:\n|status|year|code)', clean)
    if m:
        prog = m.group(1).lower()
        if any(w in prog for w in ["doctor","phd"]):   return "phd",        0.95
        if any(w in prog for w in ["master","msc"]):   return "master",     0.95
        if any(w in prog for w in ["bachelor","bsc"]): return "bachelor",   0.95

    patterns = [
        (r'\b(ph\.?d|doctor(?:ate)?)\b',               'phd',        0.95),
        (r'\b(m\.?sc|master(?:\'s)?|mba|postgrad)\b',  'master',     0.95),
        (r'\b(b\.?sc|b\.?a|bachelor(?:\'s)?|'
         r'undergraduate|b\.?tech|b\.?eng)\b',          'bachelor',   0.95),
        (r'\b(high school|secondary|gcse|waec)\b',      'high_school', 0.9),
    ]
    for pat, level, conf in patterns:
        if re.search(pat, clean):
            return level, conf

    return None, 0.0


def regex_age(text: str) -> tuple:
    tl = text.lower()

    m = re.search(r'\bage[\s:]+(\d{2})\b', tl)
    if m:
        age = int(m.group(1))
        if 13 <= age <= 35: return age, 0.9

    dob_patterns = [
        r'(?:birth\s*date|date\s*of\s*birth|dob|born)[\s:]+(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
        r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
        r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+(\d{4})',
    ]
    months = {"january":1,"february":2,"march":3,"april":4,"may":5,
               "june":6,"july":7,"august":8,"september":9,
               "october":10,"november":11,"december":12}

    for pat in dob_patterns:
        for m in re.finditer(pat, tl):
            try:
                g = m.groups()
                if g[1] in months:
                    day, month, year = int(g[0]), months[g[1]], int(g[2])
                elif len(g[0]) == 4:
                    year, month, day = int(g[0]), int(g[1]), int(g[2])
                else:
                    day, month, year = int(g[0]), int(g[1]), int(g[2])
                if not (1985 <= year <= 2010): continue
                dob   = datetime.date(year, month, day)
                today = datetime.date.today()
                age   = today.year - dob.year - (
                    (today.month, today.day) < (dob.month, dob.day)
                )
                if 13 <= age <= 35: return age, 0.88
            except (ValueError, TypeError):
                continue

    return None, 0.0


def regex_gender(text: str) -> tuple:
    tl = text.lower()
    if re.search(r'\bgender[\s:]+(?:male|man)\b',    tl): return "Male",   0.95
    if re.search(r'\bgender[\s:]+(?:female|woman)\b',tl): return "Female", 0.95
    if re.search(r'\b(mr\.?|sir)\b',  tl):               return "Male",   0.80
    if re.search(r'\b(ms\.?|mrs\.?|miss)\b', tl):        return "Female", 0.80
    return None, 0.0


def regex_name(text: str) -> tuple:
    m = re.search(
        r'student\s+name\s*[:=]\s*([A-Za-z][A-Za-z\s]{3,80}?)(?:\n|birth|student\s+id)',
        text, re.IGNORECASE
    )
    if m:
        name = m.group(1).strip()
        if not any(b in name.lower() for b in BAD_NAMES):
            return name, 0.92

    m = re.search(r'^Name:\s*([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*$',
                  text, re.MULTILINE)
    if m:
        return m.group(1).strip(), 0.85

    return None, 0.0


def regex_field(text: str) -> tuple:
    tl = text.lower()
    cut = min(
        tl.find("assessment notation") if "assessment notation" in tl else len(tl),
        tl.find("high distinction")    if "high distinction"    in tl else len(tl),
    )
    clean = tl[:cut] if cut > 100 else tl

    patterns = [
        r'award\s+programme\s*[:=]\s*([a-z\s&,]+?)(?:\n|marking|awarded)',
        r'program\s+name\s+([a-z\s]+?)(?:\n|status|year|code|active)',
        r'department\s*[:=]\s*([a-z\s&,]+?)(?:\n|,|\.|;)',
        r'bachelor(?:\'s)? (?:of|in) ([a-z\s&]+?)(?:\n|,|\.|;)',
        r'master(?:\'s)? (?:of|in) ([a-z\s&]+?)(?:\n|,|\.|;)',
    ]
    for pat in patterns:
        m = re.search(pat, clean)
        if m:
            field = m.group(1).strip().title()
            bad   = any(re.search(r'\b'+w+r'\b', field.lower())
                        for w in BAD_FIELDS)
            if 3 <= len(field) <= 80 and not bad:
                return field, 0.80

    return None, 0.0


# ══════════════════════════════════════════════════════════════
# SECTION 4 — MERGE: ROBERTA + REGEX
# ══════════════════════════════════════════════════════════════

def merge_fields(roberta_fields: dict, text: str,
                  doc_type: str = "unknown") -> dict:
    """
    Merge RoBERTa extractions with regex fallback.
    RoBERTa is used if present and confidence ≥ 0.55.
    Regex fills any remaining gap.
    For transcripts: structured fields (name, GPA) prefer regex.
    For CVs: unstructured fields prefer RoBERTa.
    """
    merged = {}

    def add(field, value, confidence, source):
        if value is not None and confidence >= 0.5:
            merged[field] = {
                "value":      value,
                "confidence": round(confidence, 3),
                "source":     source
            }

    def use_roberta(field):
        return (field in roberta_fields and
                roberta_fields[field]["confidence"] >= 0.55)

    # ── GPA ────────────────────────────────────────────────────
    # Regex is more reliable for GPA (handles module marks, scales)
    # RoBERTa used only if regex finds nothing
    gpa, gc = regex_gpa(text)
    if gpa is not None:
        add("final_gpa", gpa, gc, "regex")
    elif use_roberta("final_gpa"):
        r = roberta_fields["final_gpa"]
        add("final_gpa", r["value"], r["confidence"], "roberta")

    # ── Degree level ───────────────────────────────────────────
    # RoBERTa first for CVs, regex first for transcripts
    if doc_type == "transcript":
        deg, dc = regex_degree_level(text)
        if deg: add("degree_level", deg, dc, "regex")
        elif use_roberta("degree_level"):
            r = roberta_fields["degree_level"]
            add("degree_level", r["value"], r["confidence"], "roberta")
    else:
        if use_roberta("degree_level"):
            r = roberta_fields["degree_level"]
            add("degree_level", r["value"], r["confidence"], "roberta")
        else:
            deg, dc = regex_degree_level(text)
            add("degree_level", deg, dc, "regex")

    # ── Age ─────────────────────────────────────────────────────
    age, ac = regex_age(text)
    add("age", age, ac, "regex")

    # ── Gender ──────────────────────────────────────────────────
    gen, gnc = regex_gender(text)
    add("gender", gen, gnc, "regex")

    # ── Person name ─────────────────────────────────────────────
    # Transcripts: structured "Student Name :" → regex wins
    # CVs: free-form → RoBERTa wins
    if doc_type == "transcript":
        name, nc = regex_name(text)
        if name: add("person_name", name, nc, "regex")
        elif use_roberta("person_name"):
            r = roberta_fields["person_name"]
            add("person_name", r["value"], r["confidence"], "roberta")
    else:
        if use_roberta("person_name"):
            r = roberta_fields["person_name"]
            add("person_name", r["value"], r["confidence"], "roberta")
        else:
            name, nc = regex_name(text)
            add("person_name", name, nc, "regex")

    # ── Institution ─────────────────────────────────────────────
    # RoBERTa better here (handles multi-word institution names)
    if use_roberta("institution"):
        r = roberta_fields["institution"]
        add("institution", r["value"], r["confidence"], "roberta")
    else:
        m = re.search(r'([A-Z][a-z]+ (?:University|College|Institute)[^,\n]*)',
                      text)
        if m: add("institution", m.group(1).strip(), 0.70, "regex")

    # ── Field of study ──────────────────────────────────────────
    if use_roberta("field_of_study"):
        r = roberta_fields["field_of_study"]
        add("field_of_study", r["value"], r["confidence"], "roberta")
    else:
        fld, fc = regex_field(text)
        add("field_of_study", fld, fc, "regex")

    return merged


# ══════════════════════════════════════════════════════════════
# SECTION 5 — DOCUMENT TYPE DETECTION
# ══════════════════════════════════════════════════════════════

TRANSCRIPT_KW = ["transcript", "grade", "gpa", "cgpa", "cumulative",
                  "credit hours", "module", "marking", "degree year",
                  "academic standing", "award programme", "student mark"]
CV_KW = ["curriculum vitae", "resume", "objective", "summary",
          "experience", "employment", "internship", "skills",
          "references", "linkedin", "projects", "certifications"]


def detect_doc_type(text: str) -> str:
    tl = text.lower()
    ts = sum(1 for kw in TRANSCRIPT_KW if kw in tl)
    cs = sum(1 for kw in CV_KW if kw in tl)
    if ts >= 3 and ts > cs: return "transcript"
    if cs >= 3 and cs > ts: return "cv"
    if ts > 0 or cs > 0:   return "transcript" if ts >= cs else "cv"
    return "unknown"


# ══════════════════════════════════════════════════════════════
# SECTION 6 — MAIN PARSER
# ══════════════════════════════════════════════════════════════

def parse_document(pdf_path: str) -> dict:
    """
    Parse a student PDF. Returns structured fields with
    confidence scores and source attribution (roberta/regex).
    """
    print("\n" + "="*62)
    print("HYBRID DOCUMENT PARSER (RoBERTa NER + Regex Fallback)")
    print("="*62)

    result = {
        "success": False, "doc_type": "unknown",
        "extraction_method": "failed",
        "fields": {}, "roberta_entities": {}, "warnings": []
    }

    if not os.path.exists(pdf_path):
        result["warnings"].append(f"File not found: {pdf_path}")
        return result

    text, method = extract_text(pdf_path)
    result["extraction_method"] = method

    if not text:
        result["warnings"].append("Could not extract text.")
        return result

    result["success"]  = True
    result["doc_type"] = detect_doc_type(text)
    print(f"  Document type : {result['doc_type'].upper()}")

    print("  Running RoBERTa NER...")
    grouped         = extract_with_roberta(text)
    roberta_fields  = roberta_to_fields(grouped)
    result["roberta_entities"] = grouped
    print(f"  RoBERTa found : {list(roberta_fields.keys()) or 'nothing'}")

    print("  Applying regex fallback...")
    merged = merge_fields(roberta_fields, text, result["doc_type"])
    result["fields"] = merged

    if "final_gpa"    not in merged:
        result["warnings"].append("GPA not found — please enter manually.")
    if "degree_level" not in merged:
        result["warnings"].append("Degree level not detected — please select.")

    return result


# ══════════════════════════════════════════════════════════════
# SECTION 7 — OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════

def print_results(result: dict) -> None:
    print("\n" + "─"*65)
    print(f"  Document Type    : {result['doc_type'].upper()}")
    print(f"  Extraction Method: {result['extraction_method']}")
    print(f"  Success          : {result['success']}")

    if result["fields"]:
        print(f"\n  {'Field':<20} {'Value':<32} {'Conf':>5}  Source")
        print("  " + "─"*65)
        for f, d in result["fields"].items():
            bar = "█" * int(d["confidence"] * 10)
            src = d.get("source", "?")
            print(f"  {f:<20} {str(d['value']):<32} "
                  f"{d['confidence']:.0%}  [{src}]  {bar}")
    else:
        print("\n  No fields extracted.")

    if result["warnings"]:
        print("\n  Warnings:")
        for w in result["warnings"]:
            print(f"  ⚠  {w}")


def to_student_profile(result: dict, defaults: dict = None) -> dict:
    defaults = defaults or {
        "final_gpa": None, "degree_level": "bachelor",
        "age": 20, "gender": "Male",
        "household_income": None, "ses_category": "Middle",
        "financial_need": 0, "International": 0,
    }
    profile = dict(defaults)
    for f, d in result.get("fields", {}).items():
        if d["confidence"] >= 0.6 and f in profile:
            profile[f] = d["value"]
    profile["_doc_type"]       = result.get("doc_type")
    profile["_institution"]    = result["fields"].get("institution",    {}).get("value")
    profile["_field_of_study"] = result["fields"].get("field_of_study", {}).get("value")
    profile["_person_name"]    = result["fields"].get("person_name",    {}).get("value")
    return profile


def streamlit_confirmation_ui(result: dict) -> dict:
    """Returns pre-filled values for Streamlit confirmation widgets."""
    fields      = result.get("fields", {})
    deg_opts    = ["bachelor", "master", "phd", "high_school"]
    deg_val     = fields.get("degree_level", {}).get("value", "bachelor")
    deg_idx     = deg_opts.index(deg_val) if deg_val in deg_opts else 0
    gen_opts    = ["Male", "Female"]
    gen_val     = fields.get("gender", {}).get("value", "Male")
    gen_idx     = gen_opts.index(gen_val) if gen_val in gen_opts else 0

    return {
        "final_gpa":        fields.get("final_gpa",     {}).get("value") or 2.5,
        "degree_level":     deg_val,
        "degree_level_idx": deg_idx,
        "age":              fields.get("age",            {}).get("value") or 20,
        "gender":           gen_val,
        "gender_idx":       gen_idx,
        "field_of_study":   fields.get("field_of_study",{}).get("value") or "",
        "institution":      fields.get("institution",   {}).get("value") or "",
        "person_name":      fields.get("person_name",   {}).get("value") or "",
        "roberta_used":     any(d.get("source") == "roberta"
                                for d in fields.values()),
        "doc_type":         result.get("doc_type", "unknown"),
        "warnings":         result.get("warnings", []),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result  = parse_document(sys.argv[1])
        print_results(result)
        profile = to_student_profile(result)
        print("\n  Student Profile:")
        for k, v in profile.items():
            prefix = "  " if not k.startswith("_") else "  [meta] "
            print(f"{prefix}{k:<22}: {v}")
    else:
        print("\n[No PDF provided — run: python document_parser_hybrid.py transcript.pdf]")
