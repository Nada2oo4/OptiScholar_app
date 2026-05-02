"""
OptiScholar — AI-Driven Scholarship Recommendation System
==========================================================
Multi-page Streamlit application.

Pages:
  1. 🏠 Home
  2. 👤 My Profile (upload document or manual form)
  3. 🎯 Recommendations
  4. 🗺️ Eligibility Roadmap
  5. 💬 Chatbot

Run:
  streamlit run app.py
"""

import streamlit as st
import os
import sys

# ── Page config (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="OptiScholar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────────────
DRIVE_BASE       = "/content/drive/Graduation Project/dataset_test2"
OPTISCHOLAR_PATH = f"{DRIVE_BASE}/scholarships_final_ready-2.csv"
STUDENTS_PATH    = f"{DRIVE_BASE}/nigerian_students_v2.csv"
INTERACTIONS_PATH= f"{DRIVE_BASE}/nigerian_ncf_data_v2.csv"
TRANSFER_FN_PATH = "transfer_fn_v2.pt"
NCF_MODEL_PATH   = "ncf_model_v2.pt"

# ── Theme CSS ───────────────────────────────────────────────────
def inject_css(dark_mode: bool = False):
    if dark_mode:
        bg, surface, text, muted = "#0f1117", "#1a1d2e", "#f0f2f6", "#8b93a7"
        border = "#2d3250"
    else:
        bg, surface, text, muted = "#ffffff", "#f8f9fc", "#1a1f2e", "#6b7280"
        border = "#e5e7eb"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {{
        --primary:   #1e3a8a;
        --primary-light: #2d54c5;
        --gold:      #f59e0b;
        --gold-light:#fef3c7;
        --success:   #10b981;
        --bg:        {bg};
        --surface:   {surface};
        --text:      {text};
        --muted:     {muted};
        --border:    {border};
        --radius:    12px;
        --shadow:    0 4px 24px rgba(30,58,138,0.08);
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--primary) !important;
        border-right: none;
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.2s;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: rgba(255,255,255,0.1);
    }}

    /* Main headings */
    h1, h2, h3 {{
        font-family: 'DM Serif Display', serif !important;
        color: var(--primary);
    }}
    h1 {{ font-size: 2.4rem !important; }}
    h2 {{ font-size: 1.6rem !important; }}

    /* Cards */
    .sch-card {{
        background: var(--surface);
        border: 1.5px solid var(--border);
        border-radius: var(--radius);
        padding: 20px;
        margin-bottom: 14px;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
    }}
    .sch-card:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }}
    .sch-card.gold {{
        border-color: var(--gold);
        background: linear-gradient(135deg, var(--surface) 0%, #fffbeb 100%);
        box-shadow: 0 0 0 1px var(--gold), 0 8px 32px rgba(245,158,11,0.12);
    }}
    .sch-card.gray {{
        opacity: 0.75;
    }}

    /* Rank badge */
    .rank-badge {{
        position: absolute;
        top: -10px;
        left: 16px;
        background: var(--gold);
        color: #1a1f2e;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 20px;
        font-family: 'DM Sans', sans-serif;
    }}
    .rank-badge.blue {{
        background: var(--primary);
        color: white;
    }}

    /* Score pill */
    .score-pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .score-gold {{ background: var(--gold-light); color: #92400e; }}
    .score-blue {{ background: #dbeafe; color: var(--primary); }}

    /* Metric cards */
    .metric-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
        text-align: center;
    }}
    .metric-value {{
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: var(--primary);
    }}
    .metric-label {{
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 2px;
    }}

    /* Chat bubbles */
    .chat-user {{
        background: var(--primary);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 75%;
        float: right;
        clear: both;
        font-size: 0.92rem;
    }}
    .chat-bot {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 80%;
        float: left;
        clear: both;
        font-size: 0.92rem;
    }}

    /* Buttons */
    .stButton > button {{
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        transition: background 0.2s !important;
    }}
    .stButton > button:hover {{
        background: var(--primary-light) !important;
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background: var(--primary) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }}

    /* Hide Streamlit branding */
    #MainMenu, footer, header {{visibility: hidden;}}

    /* Logo area */
    .logo-text {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: white;
        letter-spacing: -0.5px;
    }}
    .logo-sub {{
        font-size: 0.7rem;
        color: rgba(255,255,255,0.65);
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "page":             "🏠 Home",
        "dark_mode":        False,
        "profile":          None,
        "recommendations":  None,
        "gap_results":      None,
        "opto_df":          None,
        "chatbot":          None,
        "models_loaded":    False,
        "transfer_fn":      None,
        "ncf_model":        None,
        "student_embs":     None,
        "encoders":         None,
        "sbert_model":      None,
        "sch_embs":         None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_optischolar():
    import pandas as pd
    df = pd.read_csv(OPTISCHOLAR_PATH)
    df.columns = df.columns.str.strip()
    for col in ["min_gpa_required","funding_amount_raw",
                "requires_financial_need","eligible_bachelor",
                "eligible_master","eligible_phd","eligible_high_school"]:
        df[col] = pd.to_numeric(df.get(col,0), errors="coerce").fillna(0)
    for col in ["scholarship_title","scholarship_id","description_cleaned",
                "scholarship_type","citizenship_required"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    """Load Transfer Function + NCF + SBERT models."""
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load interactions + students for encoders
    interactions = pd.read_csv(INTERACTIONS_PATH)
    students     = pd.read_csv(STUDENTS_PATH)

    SCHOLARSHIP_TYPES = ["Merit-Based","Need-Based","Academic Excellence",
                         "Community Service","Athletic"]

    le_stu    = LabelEncoder().fit(interactions["student_id"])
    le_sch    = LabelEncoder().fit(interactions["scholarship_id"])
    le_type   = LabelEncoder().fit(SCHOLARSHIP_TYPES)
    le_deg    = LabelEncoder().fit(students["degree_level"])
    le_ses    = LabelEncoder().fit(students["ses_category"])
    le_gender = LabelEncoder().fit(students["gender"])
    scaler    = MinMaxScaler().fit(
        students[["final_gpa","household_income","age"]].fillna(0)
    )

    encoders = {
        "le_stu": le_stu, "le_sch": le_sch, "le_type": le_type,
        "le_deg": le_deg, "le_ses": le_ses, "le_gender": le_gender,
        "scaler": scaler, "DEVICE": DEVICE,
        "amt_max": interactions["amount"].max()
    }

    # Load NCF model
    sys.path.insert(0, ".")
    try:
        from ncf_pipeline import NCFModel
        ncf = NCFModel(len(le_stu.classes_), len(le_sch.classes_),
                       emb_dim=8, hidden=[32,16], n_sf=6, n_cf=2,
                       dropout=0.3).to(DEVICE)
        ncf.load_state_dict(torch.load(NCF_MODEL_PATH,
                                        map_location=DEVICE))
        ncf.eval()

        # Extract student embeddings
        idx  = torch.arange(len(le_stu.classes_),
                             dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            student_embs = ncf.get_student_embedding(idx).cpu().numpy()
    except Exception as e:
        student_embs = None
        ncf          = None

    # Load Transfer Function
    try:
        from ncf_pipeline import TransferFunction
        tf = TransferFunction(in_dim=21, hidden=[128,64,32],
                               dropout=0.3).to(DEVICE)
        tf.load_state_dict(torch.load(TRANSFER_FN_PATH,
                                       map_location=DEVICE))
        tf.eval()
    except Exception as e:
        tf = None

    # Load SBERT
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sbert = None

    return ncf, tf, student_embs, encoders, sbert


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style='padding: 8px 0 24px 0'>
            <div class='logo-text'>🎓 OptiScholar</div>
            <div class='logo-sub'>AI Scholarship Advisor</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        pages = ["🏠 Home", "👤 My Profile",
                 "🎯 Recommendations", "🗺️ Eligibility Roadmap",
                 "💬 Chatbot"]
        page = st.radio("Navigation", pages,
                         index=pages.index(st.session_state.page),
                         label_visibility="collapsed")
        st.session_state.page = page

        st.divider()

        # Profile quick-view
        if st.session_state.profile:
            p = st.session_state.profile
            st.markdown("**Your Profile**")
            gpa    = p.get("final_gpa", "—")
            degree = p.get("degree_level", "—")
            need   = "✅" if p.get("financial_need") else "❌"
            st.markdown(f"GPA: **{gpa}** | {degree.title()}")
            st.markdown(f"Financial need: {need}")
            if st.button("✏️ Edit Profile", use_container_width=True):
                st.session_state.page = "👤 My Profile"
                st.rerun()

        st.divider()

        # Dark mode toggle
        dark = st.toggle("🌙 Dark Mode",
                          value=st.session_state.dark_mode)
        st.session_state.dark_mode = dark

        # System status
        st.markdown("**System Status**")
        status_items = {
            "OptiScholar DB": os.path.exists(OPTISCHOLAR_PATH),
            "Transfer Model": os.path.exists(TRANSFER_FN_PATH),
            "NCF Model":      os.path.exists(NCF_MODEL_PATH),
        }
        for name, ok in status_items.items():
            icon = "🟢" if ok else "🔴"
            st.markdown(f"{icon} {name}")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════

def page_home():
    st.markdown("# OptiScholar")
    st.markdown("#### *AI-Driven Scholarship Recommendation System*")
    st.markdown("---")

    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("11,289", "Scholarships"),
        ("82.4%", "Model AUC"),
        ("5", "DL Models"),
        ("4", "Eligibility Levels"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    features = [
        ("🎯", "Smart Matching",
         "Transfer Learning matches your profile to 11k+ international scholarships using Neural Collaborative Filtering."),
        ("🗺️", "Eligibility Roadmap",
         "See exactly which scholarships you're almost qualifying for and what specific actions to take."),
        ("📄", "Document Parser",
         "Upload your transcript or CV — DeepSeek-OCR-2 + BERT NER extract your profile automatically."),
    ]
    for col, (icon, title, desc) in zip([col1,col2,col3], features):
        with col:
            st.markdown(f"""
            <div class='sch-card'>
                <div style='font-size:2rem;margin-bottom:8px'>{icon}</div>
                <div style='font-family:"DM Serif Display",serif;
                            font-size:1.1rem;color:var(--primary);
                            margin-bottom:6px'>{title}</div>
                <div style='font-size:0.88rem;color:var(--muted);
                            line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System architecture
    st.markdown("### System Architecture")
    st.markdown("""
    ```
    Student Profile / Document Upload
              ↓
    Document Parser (DeepSeek-OCR-2 + BERT NER + Regex)
              ↓
    Eligibility Filter (K-Mapping: 11,289 → ~8,031)
              ↓
    ┌─────────────────────────────────────┐
    │  NCF + Attention    BiLSTM (text)   │
    │  (collaborative)    SBERT (semantic)│
    └──────────────┬──────────────────────┘
                   ↓
         Transfer Function (AUC 82.4%) ← Main DL Component
                   ↓
    Ranked Recommendations + Gap Finder Roadmap
    ```
    """)

    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "👤 My Profile"
            st.rerun()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MY PROFILE
# ══════════════════════════════════════════════════════════════

def page_profile():
    st.markdown("# 👤 My Profile")
    st.markdown("Fill in your details to get personalised scholarship recommendations.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Upload Document", "✏️ Manual Entry"])

    # ── Tab 1: Document Upload ──────────────────────────────────
    with tab1:
        st.markdown("### Upload your Transcript or CV")
        st.info("Supported: PDF files (typed or scanned). "
                "Fields will be extracted automatically and shown for confirmation.")

        uploaded = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="doc_upload"
        )

        if uploaded:
            import tempfile
            with st.spinner("🔍 Extracting your profile with AI..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    # Try hybrid parser
                    try:
                        from document_parser_hybrid import (
                            parse_document, print_results,
                            streamlit_confirmation_ui
                        )
                        result  = parse_document(tmp_path)
                        prefill = streamlit_confirmation_ui(result)
                        extraction_ok = result["success"]
                    except Exception as e:
                        st.warning(f"Parser error: {e}. Please fill manually.")
                        prefill = {}
                        extraction_ok = False

                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    prefill = {}
                    extraction_ok = False

            if extraction_ok:
                # Show extraction summary
                fields = result.get("fields", {})
                if fields:
                    st.success(
                        f"✅ Extracted {len(fields)} fields from your "
                        f"{result.get('doc_type','document').upper()}. "
                        "Please review and confirm below."
                    )
                    cols = st.columns(len(fields))
                    for col, (fname, fdata) in zip(cols, fields.items()):
                        with col:
                            src = fdata.get("source", "?")
                            conf = fdata.get("confidence", 0)
                            icon = "🤖" if src in ["ner","roberta"] else "📐"
                            st.metric(
                                label=f"{icon} {fname.replace('_',' ').title()}",
                                value=str(fdata["value"]),
                                delta=f"{conf:.0%} confidence"
                            )

                st.markdown("**Confirm or edit extracted fields:**")
            else:
                st.warning("Could not auto-extract fields. Please fill manually.")
                prefill = {}

            # Show confirmation form
            _render_profile_form(prefill)

    # ── Tab 2: Manual Entry ─────────────────────────────────────
    with tab2:
        st.markdown("### Enter your details manually")
        _render_profile_form({})


def _render_profile_form(prefill: dict):
    """Shared profile form used by both tabs."""
    with st.form("profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Academic Information**")
            gpa = st.number_input(
                "GPA (0.0 – 4.0)",
                min_value=0.0, max_value=4.0, step=0.01,
                value=float(prefill.get("final_gpa") or 2.5),
                format="%.2f"
            )
            degree_opts = ["bachelor","master","phd","high_school"]
            degree = st.selectbox(
                "Degree Level",
                degree_opts,
                index=prefill.get("degree_level_idx", 0)
            )
            field = st.text_input(
                "Field of Study",
                value=str(prefill.get("field_of_study") or "")
            )
            institution = st.text_input(
                "Institution",
                value=str(prefill.get("institution") or "")
            )

        with col2:
            st.markdown("**Personal Information**")
            age = st.number_input(
                "Age", min_value=13, max_value=35, step=1,
                value=int(prefill.get("age") or 20)
            )
            gender = st.selectbox(
                "Gender",
                ["Male","Female"],
                index=prefill.get("gender_idx", 0)
            )
            ses = st.selectbox(
                "Socioeconomic Status",
                ["Low","Middle","High"], index=1
            )
            income = st.number_input(
                "Household Income (USD/year)",
                min_value=0, max_value=3000000, step=1000,
                value=50000
            )

        st.markdown("**Eligibility**")
        col3, col4 = st.columns(2)
        with col3:
            financial_need = st.checkbox("I demonstrate financial need")
        with col4:
            international  = st.checkbox("I am an international student")

        submitted = st.form_submit_button(
            "💾 Save Profile & Find Scholarships",
            use_container_width=True
        )

        if submitted:
            st.session_state.profile = {
                "final_gpa":        gpa,
                "degree_level":     degree,
                "study_level":      degree,
                "gpa_proxy":        gpa,
                "age":              age,
                "gender":           gender,
                "ses_category":     ses,
                "household_income": income,
                "financial_need":   int(financial_need),
                "International":    int(international),
                "_field_of_study":  field,
                "_institution":     institution,
            }
            # Reset recommendations when profile changes
            st.session_state.recommendations = None
            st.session_state.gap_results     = None

            st.success("✅ Profile saved! Go to Recommendations to find your scholarships.")
            st.balloons()


# ══════════════════════════════════════════════════════════════
# PAGE 3 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════

def page_recommendations():
    st.markdown("# 🎯 Scholarship Recommendations")
    st.markdown("---")

    if not st.session_state.profile:
        st.warning("⚠️ Please complete your profile first.")
        if st.button("Go to Profile"):
            st.session_state.page = "👤 My Profile"
            st.rerun()
        return

    profile = st.session_state.profile
    opto_df = st.session_state.opto_df

    # Profile summary
    with st.expander("📋 Your Profile", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GPA", f"{profile.get('final_gpa',0):.2f}")
        col2.metric("Level", profile.get("degree_level","—").title())
        col3.metric("Financial Need",
                    "Yes" if profile.get("financial_need") else "No")
        col4.metric("International",
                    "Yes" if profile.get("International") else "No")

    # Find scholarships button
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        top_n = st.selectbox("Show top", [10,20,30,50], index=1)
    with col2:
        run = st.button("🔍 Find Scholarships", use_container_width=True)

    if run or st.session_state.recommendations is not None:
        if run:
            with st.spinner("🤖 Running Transfer Learning model..."):
                try:
                    recs = _run_recommendations(profile, opto_df, top_n)
                    st.session_state.recommendations = recs
                except Exception as e:
                    st.error(f"Recommendation error: {e}")
                    return

        recs = st.session_state.recommendations
        if recs is None or len(recs) == 0:
            st.warning("No scholarships found for your profile. "
                       "Try adjusting your eligibility settings.")
            return

        # Update chatbot context
        if st.session_state.chatbot:
            st.session_state.chatbot.set_context(
                recommendations=recs, opto_df=opto_df
            )

        st.markdown(f"### Found **{len(recs):,}** matching scholarships")
        st.markdown("*Ranked by Transfer Learning match score*")
        st.markdown("<br>", unsafe_allow_html=True)

        # Scholarship cards
        for i, (_, row) in enumerate(recs.iterrows()):
            score  = float(row.get("transfer_score", 0) or 0)
            title  = str(row.get("scholarship_title",""))
            stype  = str(row.get("scholarship_type",""))
            amount = float(row.get("funding_amount_raw",0) or 0)
            desc   = str(row.get("description_cleaned",""))[:200]

            # Visual tier
            if i < 3:
                card_cls  = "sch-card gold"
                badge_cls = "rank-badge"
                score_cls = "score-pill score-gold"
                badge_txt = f"⭐ #{i+1} Best Match"
            elif i < 10:
                card_cls  = "sch-card"
                badge_cls = "rank-badge blue"
                score_cls = "score-pill score-blue"
                badge_txt = f"#{i+1}"
            else:
                card_cls  = "sch-card gray"
                badge_cls = "rank-badge blue"
                score_cls = "score-pill score-blue"
                badge_txt = f"#{i+1}"

            st.markdown(f"""
            <div class='{card_cls}'>
                <span class='{badge_cls}'>{badge_txt}</span>
                <div style='margin-top:8px;display:flex;
                            justify-content:space-between;
                            align-items:flex-start'>
                    <div style='flex:1'>
                        <div style='font-family:"DM Serif Display",serif;
                                    font-size:1.05rem;color:var(--primary);
                                    margin-bottom:4px'>{title}</div>
                        <div style='font-size:0.83rem;color:var(--muted);
                                    margin-bottom:8px'>{desc}{'...' if len(desc)==200 else ''}</div>
                        <span style='background:#e0e7ff;color:#3730a3;
                                     padding:2px 8px;border-radius:4px;
                                     font-size:0.75rem;margin-right:6px'>{stype}</span>
                    </div>
                    <div style='text-align:right;min-width:120px;
                                margin-left:16px'>
                        <div style='font-family:"DM Serif Display",serif;
                                    font-size:1.4rem;color:var(--primary)'>
                            ${amount:,.0f}</div>
                        <div class='{score_cls}'>
                            Match: {score:.0%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _run_recommendations(profile, opto_df, top_n=20):
    """Run the Transfer Function recommendation pipeline."""
    import torch
    import numpy as np

    # Load models
    ncf, tf, student_embs, encoders, sbert = load_models()
    if tf is None:
        raise ValueError("Transfer Function model not loaded.")

    DEVICE  = encoders["DEVICE"]
    le_type = encoders["le_type"]
    le_ses  = encoders["le_ses"]
    le_deg  = encoders["le_deg"]
    le_gender = encoders["le_gender"]
    scaler  = encoders["scaler"]
    amt_max = encoders["amt_max"]

    # Get proxy student embedding
    gpa    = float(profile.get("gpa_proxy") or profile.get("final_gpa") or 2.5)
    income = float(profile.get("household_income", 50000))
    age    = float(profile.get("age", 20))

    cont  = scaler.transform([[gpa, income, age]])[0]
    gpa_n = cont[0]

    n = len(student_embs)
    if gpa_n >= 0.66:    seg = student_embs[int(n*0.66):]
    elif gpa_n >= 0.33:  seg = student_embs[int(n*0.33):int(n*0.66)]
    else:                seg = student_embs[:int(n*0.33)]
    stu_emb = seg.mean(axis=0)

    # Eligibility filter
    level     = profile.get("degree_level", "bachelor")
    level_col = f"eligible_{level}"
    if level_col not in opto_df.columns:
        level_col = "eligible_bachelor"

    intl  = int(profile.get("International", 0))
    cands = opto_df[opto_df[level_col] == 1].copy()
    cands = cands[
        (cands["min_gpa_required"] <= 0) |
        (cands["min_gpa_required"] <= gpa * 5.0)
    ]
    if intl:
        cands = cands[
            ~cands["citizenship_required"].str.contains(
                "us_citizen|specific_residency", case=False, na=False
            )
        ]

    if cands.empty:
        return cands

    # Build feature matrix
    def safe_type(t):
        t = str(t).strip()
        return le_type.transform([t])[0] if t in le_type.classes_ else 0

    X_list = []
    for _, row in cands.iterrows():
        type_enc = safe_type(row.get("scholarship_type","Merit-Based"))
        amount_n = float(row.get("funding_amount_raw",0)) / (amt_max+1e-8)
        deg_enc  = 1 if level == "bachelor" else 0
        sch_vec  = np.array([type_enc, amount_n, gpa_n,
                              cont[1], deg_enc], dtype=float)
        X_list.append(np.concatenate([stu_emb, sch_vec]))

    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(DEVICE)
    tf.eval()
    with torch.no_grad():
        scores = tf(X).cpu().numpy()

    cands = cands.copy().reset_index(drop=True)
    cands["transfer_score"] = scores
    return cands.sort_values("transfer_score", ascending=False).head(top_n)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ELIGIBILITY ROADMAP
# ══════════════════════════════════════════════════════════════

def page_roadmap():
    st.markdown("# 🗺️ Eligibility Roadmap")
    st.markdown("Scholarships you're close to qualifying for — "
                "with specific actions to close the gap.")
    st.markdown("---")

    if not st.session_state.profile:
        st.warning("⚠️ Please complete your profile first.")
        return

    profile = st.session_state.profile
    opto_df = st.session_state.opto_df

    col1, col2 = st.columns([1,3])
    with col1:
        top_n = st.selectbox("Show top", [5,10,15,20], index=1)
    with col2:
        run = st.button("🗺️ Find My Eligibility Gaps",
                         use_container_width=True)

    if run or st.session_state.gap_results is not None:
        if run:
            with st.spinner("🔍 Scanning for near-miss scholarships..."):
                try:
                    gaps = _run_gap_finder(profile, opto_df, top_n)
                    st.session_state.gap_results = gaps
                except Exception as e:
                    st.error(f"Gap Finder error: {e}")
                    return

        gaps = st.session_state.gap_results

        if st.session_state.chatbot:
            st.session_state.chatbot.set_context(gap_results=gaps)

        if gaps is None or len(gaps) == 0:
            st.success("🎉 Great news! You qualify for all nearby scholarships. "
                       "Check the Recommendations page for your matches.")
            return

        st.markdown(f"### Found **{len(gaps)}** scholarships you're close to qualifying for")
        st.markdown("<br>", unsafe_allow_html=True)

        for i, (_, row) in enumerate(gaps.iterrows(), 1):
            title   = str(row.get("scholarship_title",""))
            stype   = str(row.get("scholarship_type",""))
            amount  = float(row.get("funding_amount",0) or 0)
            gap     = float(row.get("gap_score",0) or 0)
            actions = str(row.get("action_items", row.get("gaps","")))

            # Gap score → color
            if gap <= 0.4:
                border_color = "#10b981"
                label        = "🟢 Very Close"
            elif gap <= 0.8:
                border_color = "#f59e0b"
                label        = "🟡 Close"
            else:
                border_color = "#ef4444"
                label        = "🔴 Some Work Needed"

            st.markdown(f"""
            <div class='sch-card' style='border-left:4px solid {border_color}'>
                <div style='display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:10px'>
                    <div style='font-family:"DM Serif Display",serif;
                                font-size:1.05rem;color:var(--primary)'>
                        {i}. {title}</div>
                    <div>{label}</div>
                </div>
                <div style='display:flex;gap:16px;margin-bottom:10px;
                            flex-wrap:wrap'>
                    <span style='font-size:0.83rem;color:var(--muted)'>
                        💰 ${amount:,.0f}</span>
                    <span style='font-size:0.83rem;color:var(--muted)'>
                        🏷️ {stype}</span>
                    <span style='font-size:0.83rem;color:var(--muted)'>
                        📊 Gap score: {gap:.2f}</span>
                </div>
                <div style='background:#f0fdf4;border-radius:8px;
                            padding:10px 14px;font-size:0.87rem;
                            color:#166534'>
                    ✅ <strong>Action:</strong> {actions}
                </div>
            </div>
            """, unsafe_allow_html=True)


def _run_gap_finder(profile, opto_df, top_n=10):
    """Run the Gap Finder on OptiScholar scholarships."""
    import re

    gpa   = float(profile.get("final_gpa") or profile.get("gpa_proxy") or 0)
    need  = int(profile.get("financial_need", 0))
    intl  = int(profile.get("International", 0))
    level = profile.get("degree_level", "bachelor")
    lc    = f"eligible_{level}"
    if lc not in opto_df.columns:
        lc = "eligible_bachelor"

    gaps = []
    for _, sch in opto_df.iterrows():
        if not int(sch.get(lc, 0) or 0):
            continue

        items, score = [], 0.0

        min_gpa = float(sch.get("min_gpa_required", -1) or -1)
        gpa_20  = gpa * 5.0
        if min_gpa > 0 and gpa_20 < min_gpa:
            shortfall = min_gpa - gpa_20
            if shortfall > 3.0:
                continue
            items.append(
                f"Raise GPA by {shortfall/5:.2f} pts "
                f"(need {min_gpa/5:.1f}, have {gpa:.2f})"
            )
            score += shortfall / 3.0

        req_need = int(sch.get("requires_financial_need", 0) or 0)
        if req_need and not need:
            items.append("Provide financial need documentation")
            score += 0.4

        cit = str(sch.get("citizenship_required", "none") or "none").lower()
        if intl and ("us_citizen" in cit or "specific_residency" in cit):
            items.append("Requires US citizenship / specific residency")
            score += 0.8

        if items and score <= 1.5:
            gaps.append({
                "scholarship_id":    sch["scholarship_id"],
                "scholarship_title": sch["scholarship_title"],
                "scholarship_type":  sch.get("scholarship_type",""),
                "funding_amount":    float(sch["funding_amount_raw"] or 0),
                "gap_score":         round(score, 3),
                "action_items":      " | ".join(items),
            })

    if not gaps:
        return None

    import pandas as pd
    return pd.DataFrame(gaps).sort_values(
        ["gap_score","funding_amount"], ascending=[True,False]
    ).head(top_n)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — CHATBOT
# ══════════════════════════════════════════════════════════════

def page_chatbot():
    st.markdown("# 💬 OptiScholar Assistant")
    st.markdown("---")

    # Init chatbot
    if st.session_state.chatbot is None:
        from chatbot import OptiScholarChatbot
        st.session_state.chatbot = OptiScholarChatbot()

    bot = st.session_state.chatbot

    # Sync context
    bot.set_context(
        profile=st.session_state.profile,
        recommendations=st.session_state.recommendations,
        gap_results=st.session_state.gap_results,
        opto_df=st.session_state.opto_df,
    )

    # Suggested questions
    if len(bot.history) == 0:
        st.markdown("**Quick questions to get started:**")
        suggestions = bot.get_suggested_questions()
        cols = st.columns(len(suggestions))
        for col, s in zip(cols, suggestions):
            with col:
                if st.button(s, use_container_width=True,
                             key=f"sug_{s[:15]}"):
                    bot.respond(s)
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    chat_area = st.container()
    with chat_area:
        if not bot.history:
            from chatbot import GREET_RESPONSES
            import random
            st.markdown(f"""
            <div class='chat-bot'>🎓 {random.choice(GREET_RESPONSES)}</div>
            <div style='clear:both'></div>
            """, unsafe_allow_html=True)
        else:
            for role, msg in bot.history:
                if role == "user":
                    st.markdown(f"""
                    <div class='chat-user'>{msg}</div>
                    <div style='clear:both'></div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-bot'>🎓 {msg}</div>
                    <div style='clear:both'></div>
                    """, unsafe_allow_html=True)

    # Input
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([5,1])
    with col1:
        user_input = st.text_input(
            "Message", placeholder="Ask about scholarships...",
            label_visibility="collapsed", key="chat_input"
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_input.strip():
        bot.respond(user_input.strip())
        st.rerun()

    if bot.history:
        if st.button("🗑️ Clear conversation"):
            bot.clear_history()
            st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    init_session()
    inject_css(st.session_state.dark_mode)

    # Load OptiScholar data once
    if st.session_state.opto_df is None:
        if os.path.exists(OPTISCHOLAR_PATH):
            with st.spinner("Loading scholarship database..."):
                st.session_state.opto_df = load_optischolar()

    # Sidebar
    render_sidebar()

    # Route to page
    page = st.session_state.page
    if page == "🏠 Home":
        page_home()
    elif page == "👤 My Profile":
        page_profile()
    elif page == "🎯 Recommendations":
        page_recommendations()
    elif page == "🗺️ Eligibility Roadmap":
        page_roadmap()
    elif page == "💬 Chatbot":
        page_chatbot()


if __name__ == "__main__":
    main()
