"""
Neural Collaborative Filtering (NCF) + Transfer Learning
=========================================================
Phase 1: Train NCF on Nigerian v2 interaction data
Phase 2: Extract student embeddings (64-dim)
Phase 3: Train transfer function on Nigerian data
Phase 4: Apply transfer function to OptiScholar scholarships

Training data : nigerian_ncf_data_v2.csv  (19,947 interactions)
Student data  : nigerian_students_v2.csv  (8,000 students)
Target data   : scholarships_final_ready.csv (11,289 scholarships)

Bridge: scholarship_type shared across both datasets (5 categories)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
NGA_INTERACTIONS = "/content/drive/MyDrive/Graduation Project/dataset_test2/nigerian_ncf_data_v2.csv"
NGA_STUDENTS     = "/content/drive/MyDrive/Graduation Project/dataset_test2/nigerian_students_v2.csv"
OPTISCHOLAR      = "/content/drive/MyDrive/Graduation Project/dataset_test2/scholarships_final_ready-2.csv"

OUT_DIR = "figures_ncf"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SCHOLARSHIP_TYPES = [
    "Merit-Based", "Need-Based", "Academic Excellence",
    "Community Service", "Athletic"
]

COLOR_POS = "#4C72B0"
COLOR_NEG = "#DD8452"
COLOR_ACC = "#55A868"
COLOR_4   = "#8172B2"

def _save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_data():
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    interactions = pd.read_csv(NGA_INTERACTIONS)
    students     = pd.read_csv(NGA_STUDENTS)
    opto         = pd.read_csv(OPTISCHOLAR)
    opto.columns = opto.columns.str.strip()

    # Normalise OptiScholar numeric cols
    for col in ["min_gpa_required", "funding_amount_raw",
                "requires_financial_need", "has_gpa_requirement",
                "eligible_bachelor", "eligible_master",
                "eligible_phd", "eligible_high_school"]:
        opto[col] = pd.to_numeric(opto.get(col, 0),
                                   errors="coerce").fillna(0)
    for col in ["citizenship_required", "scholarship_title",
                "scholarship_id", "description_cleaned",
                "scholarship_type"]:
        if col in opto.columns:
            opto[col] = opto[col].fillna("unknown").astype(str)

    # Merge interactions with student features
    merged = interactions.merge(students, on="student_id", how="inner")
    merged["income_log"] = np.log1p(merged["household_income"])

    print(f"  Interactions : {len(interactions):,}")
    print(f"  Students     : {len(students):,}")
    print(f"  Merged rows  : {len(merged):,}")
    print(f"  OptiScholar  : {len(opto):,} scholarships")
    print(f"  Target dist  :\n{merged['target'].value_counts()}")

    return merged, students, opto


# ══════════════════════════════════════════════════════════════
# SECTION 2 — ENCODERS & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def build_encoders(merged, students):
    le_stu    = LabelEncoder().fit(merged["student_id"])
    le_sch    = LabelEncoder().fit(merged["scholarship_id"])
    le_type   = LabelEncoder().fit(SCHOLARSHIP_TYPES)
    le_deg    = LabelEncoder().fit(students["degree_level"])
    le_ses    = LabelEncoder().fit(students["ses_category"])
    le_gender = LabelEncoder().fit(students["gender"])

    scaler = MinMaxScaler().fit(
        students[["final_gpa", "household_income", "age"]].fillna(0)
    )
    return le_stu, le_sch, le_type, le_deg, le_ses, le_gender, scaler


def encode_features(merged, le_stu, le_sch, le_type,
                    le_deg, le_ses, le_gender, scaler):
    df = merged.copy()

    df["stu_idx"]  = le_stu.transform(df["student_id"])
    df["sch_idx"]  = le_sch.transform(df["scholarship_id"])

    # Scholarship type — safe encode
    df["type_enc"] = df["scholarship_type"].apply(
        lambda t: le_type.transform([t])[0]
        if t in le_type.classes_ else 0
    )

    # Amount normalised
    amt_max        = df["amount"].max()
    df["amount_n"] = df["amount"] / (amt_max + 1e-8)

    # Student features
    df["deg_enc"]    = le_deg.transform(df["degree_level"])
    df["ses_enc"]    = le_ses.transform(df["ses_category"])
    df["gender_enc"] = le_gender.transform(df["gender"])

    cont = scaler.transform(
        df[["final_gpa", "household_income", "age"]].fillna(0)
    )
    df[["gpa_n", "inc_n", "age_n"]] = cont

    return df


# ══════════════════════════════════════════════════════════════
# SECTION 3 — NEGATIVE SAMPLING
# ══════════════════════════════════════════════════════════════

def negative_sample(merged, n_neg=3, seed=42):
    np.random.seed(seed)
    all_sch  = merged["scholarship_id"].unique().tolist()
    pos      = merged[merged["target"] == 1].copy()
    seen_map = merged.groupby("student_id")["scholarship_id"]\
                     .apply(set).to_dict()

    neg_rows = []
    for _, row in pos.iterrows():
        sid  = row["student_id"]
        pool = [s for s in all_sch if s not in seen_map.get(sid, set())]
        if not pool:
            continue
        for sch in np.random.choice(
            pool, size=min(n_neg, len(pool)), replace=False
        ):
            r = row.copy()
            r["scholarship_id"] = sch
            r["target"]         = 0
            # Use scholarship features from existing rows
            sch_row = merged[merged["scholarship_id"] == sch].iloc[0]
            r["scholarship_type"] = sch_row["scholarship_type"]
            r["amount"]           = sch_row["amount"]
            neg_rows.append(r)

    combined = pd.concat(
        [merged, pd.DataFrame(neg_rows)], ignore_index=True
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n  After neg sampling ({n_neg}:1):")
    print(f"  Positive: {(combined['target']==1).sum():,}")
    print(f"  Negative: {(combined['target']==0).sum():,}")
    print(f"  Total   : {len(combined):,}")
    return combined


# ══════════════════════════════════════════════════════════════
# SECTION 4 — NCF MODEL
# ══════════════════════════════════════════════════════════════

class NCFDataset(Dataset):
    def __init__(self, df):
        self.stu  = torch.tensor(df["stu_idx"].values,  dtype=torch.long)
        self.sch  = torch.tensor(df["sch_idx"].values,  dtype=torch.long)
        # Student features: gpa, income, age, ses, deg, gender
        self.sf   = torch.tensor(
            df[["gpa_n","inc_n","age_n","ses_enc","deg_enc","gender_enc"]]
            .fillna(0).values, dtype=torch.float32
        )
        # Scholarship features: type, amount
        self.cf   = torch.tensor(
            df[["type_enc","amount_n"]].fillna(0).values,
            dtype=torch.float32
        )
        self.y    = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.stu[i], self.sch[i], self.sf[i], self.cf[i], self.y[i]


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering.
    GMF path : element-wise product of user & item embeddings
    MLP path : concatenated embeddings + side features through hidden layers
    Output   : sigmoid probability of interaction

    Student embedding (GMF 16-dim + MLP 16-dim = 32-dim) is
    extracted after training for transfer to OptiScholar.
    """
    def __init__(self, n_stu, n_sch, emb_dim=16,
                 hidden=[64, 32], n_sf=6, n_cf=2, dropout=0.3):
        super().__init__()
        self.emb_dim = emb_dim

        # GMF embeddings
        self.gmf_stu = nn.Embedding(n_stu, emb_dim)
        self.gmf_sch = nn.Embedding(n_sch, emb_dim)

        # MLP embeddings
        self.mlp_stu = nn.Embedding(n_stu, emb_dim)
        self.mlp_sch = nn.Embedding(n_sch, emb_dim)

        # MLP tower: input = 2*emb + student_feats + sch_feats
        mlp_in = 2 * emb_dim + n_sf + n_cf
        layers = []
        for h in hidden:
            layers += [nn.Linear(mlp_in, h), nn.ReLU(),
                       nn.Dropout(dropout), nn.BatchNorm1d(h)]
            mlp_in = h
        self.mlp = nn.Sequential(*layers)

        # Output: GMF(emb_dim) + MLP(hidden[-1]) -> 1
        self.out = nn.Linear(emb_dim + hidden[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_stu, self.gmf_sch,
                    self.mlp_stu, self.mlp_sch]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, stu, sch, sf, cf):
        # GMF path
        gmf = self.gmf_stu(stu) * self.gmf_sch(sch)

        # MLP path
        mlp_in = torch.cat(
            [self.mlp_stu(stu), self.mlp_sch(sch), sf, cf], dim=1
        )
        mlp = self.mlp(mlp_in)

        out = self.out(torch.cat([gmf, mlp], dim=1))
        return out.squeeze()

    def get_student_embedding(self, stu_idx):
        """Return 32-dim student embedding for transfer learning."""
        return torch.cat(
            [self.gmf_stu(stu_idx), self.mlp_stu(stu_idx)], dim=1
        )


# ══════════════════════════════════════════════════════════════
# SECTION 5 — PHASE 1: TRAIN NCF
# ══════════════════════════════════════════════════════════════

def train_ncf(data_aug, le_stu, le_sch,
              epochs=100, batch_size=512, lr=0.001, wd=1e-4,
              emb_dim=8, hidden=[32, 16], dropout=0.4):
    print("\n" + "="*60)
    print("PHASE 1 — TRAINING NCF")
    print("="*60)

    X_tr, X_te = train_test_split(
        data_aug, test_size=0.2, random_state=42,
        stratify=data_aug["target"]
    )

    tr_ld = DataLoader(NCFDataset(X_tr), batch_size=batch_size,
                       shuffle=True,  num_workers=0)
    te_ld = DataLoader(NCFDataset(X_te), batch_size=batch_size,
                       shuffle=False, num_workers=0)

    n_stu = len(le_stu.classes_)
    n_sch = len(le_sch.classes_)

    model = NCFModel(n_stu, n_sch, emb_dim=emb_dim,
                     hidden=hidden, n_sf=6,
                     n_cf=2, dropout=dropout).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"  NCF parameters : {total_params:,}")
    print(f"  Train samples  : {len(X_tr):,}")
    print(f"  Test samples   : {len(X_te):,}")

    # Weighted loss for class imbalance
    pos_w = torch.tensor(
        [(data_aug["target"]==0).sum() /
         (data_aug["target"]==1).sum()],
        dtype=torch.float32
    ).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=8, factor=0.5
    )

    best_auc, best_state = 0.0, None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'AUC':>8}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tl = 0.0
        for stu, sch, sf, cf, y in tr_ld:
            stu, sch, sf, cf, y = (stu.to(DEVICE), sch.to(DEVICE),
                                    sf.to(DEVICE), cf.to(DEVICE),
                                    y.to(DEVICE))
            opt.zero_grad()
            pred = model(stu, sch, sf, cf)
            loss = crit(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(tr_ld)

        # Validate
        model.eval()
        vl, preds, trues = 0.0, [], []
        with torch.no_grad():
            for stu, sch, sf, cf, y in te_ld:
                stu, sch, sf, cf, y = (stu.to(DEVICE), sch.to(DEVICE),
                                        sf.to(DEVICE), cf.to(DEVICE),
                                        y.to(DEVICE))
                pred  = model(stu, sch, sf, cf)
                vl   += crit(pred, y).item()
                preds.extend(pred.cpu().numpy())
                trues.extend(y.cpu().numpy())

        vl   /= len(te_ld)
        vauc  = roc_auc_score(trues, preds)
        sched.step(vl)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_auc"].append(vauc)

        if vauc > best_auc:
            best_auc   = vauc
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"{epoch:>6} {tl:>12.4f} {vl:>10.4f} {vauc:>8.4f}")

    model.load_state_dict(best_state)
    print(f"\n  Best NCF AUC : {best_auc:.4f}")

    # Final test metrics
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for stu, sch, sf, cf, y in te_ld:
            pred = model(stu.to(DEVICE), sch.to(DEVICE),
                         sf.to(DEVICE), cf.to(DEVICE))
            preds.extend(pred.cpu().numpy())
            trues.extend(y.numpy())

    preds_b = (np.array(preds) >= 0.5).astype(int)
    final_metrics = {
        "auc":       roc_auc_score(trues, preds),
        "accuracy":  accuracy_score(trues, preds_b),
        "precision": precision_score(trues, preds_b, zero_division=0),
        "recall":    recall_score(trues, preds_b, zero_division=0),
        "f1":        f1_score(trues, preds_b, zero_division=0),
    }

    print("\n  Final Test Metrics:")
    for k, v in final_metrics.items():
        print(f"    {k:<12}: {v:.4f}")

    return model, history, final_metrics, np.array(preds), np.array(trues)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — PHASE 2: EXTRACT STUDENT EMBEDDINGS
# ══════════════════════════════════════════════════════════════

def extract_embeddings(model, le_stu):
    print("\n" + "="*60)
    print("PHASE 2 — EXTRACTING STUDENT EMBEDDINGS")
    print("="*60)

    model.eval()
    idx = torch.arange(len(le_stu.classes_),
                       dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        embs = model.get_student_embedding(idx).cpu().numpy()

    print(f"  Shape: {embs.shape}  "
          f"({len(le_stu.classes_):,} students × {embs.shape[1]} dims)")
    return embs


# ══════════════════════════════════════════════════════════════
# SECTION 7 — PHASE 3: TRANSFER FUNCTION
# ══════════════════════════════════════════════════════════════

class TransferDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TransferFunction(nn.Module):
    """
    Input : student_embedding(32) + scholarship_features(5) = 37-dim
    Output: match probability

    Trained on Nigerian data, applied directly to OptiScholar.
    The shared scholarship_type space (5 categories) is the transfer bridge.
    """
    def __init__(self, in_dim=37, hidden=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(),
                       nn.Dropout(dropout), nn.BatchNorm1d(h)]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden[-1], 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.out(self.net(x)).squeeze()


def train_transfer_function(data_enc, student_embs, le_stu, le_type,
                             epochs=80, batch_size=512,
                             lr=0.001, wd=1e-4):
    print("\n" + "="*60)
    print("PHASE 3 — TRAINING TRANSFER FUNCTION")
    print("="*60)

    amt_max = data_enc["amount"].max()

    X_rows, y_rows = [], []
    for _, row in data_enc.iterrows():
        try:
            stu_idx = le_stu.transform([row["student_id"]])[0]
        except Exception:
            continue

        stu_emb = student_embs[stu_idx]

        type_enc = row["type_enc"]
        amount_n = row["amount"] / (amt_max + 1e-8)
        gpa_n    = row["gpa_n"]
        inc_n    = row["inc_n"]
        deg_enc  = row["deg_enc"]

        sch_vec = np.array([type_enc, amount_n, gpa_n, inc_n, deg_enc],
                            dtype=float)
        X_rows.append(np.concatenate([stu_emb, sch_vec]))
        y_rows.append(float(row["target"]))

    X = np.array(X_rows)
    y = np.array(y_rows)
    print(f"  Transfer samples: {len(X):,}")
    print(f"  Input dim       : {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tr_ld = DataLoader(TransferDataset(X_tr, y_tr),
                       batch_size=batch_size, shuffle=True)
    te_ld = DataLoader(TransferDataset(X_te, y_te),
                       batch_size=batch_size, shuffle=False)

    model = TransferFunction(
        in_dim=X.shape[1], hidden=[128, 64, 32], dropout=0.3
    ).to(DEVICE)

    pos_w = torch.tensor(
        [(y_tr==0).sum() / (y_tr==1).sum() + 1e-8],
        dtype=torch.float32
    ).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=8, factor=0.5
    )

    best_auc, best_state = 0.0, None
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'AUC':>8}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(tr_ld)

        model.eval()
        vl, preds, trues = 0.0, [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred    = model(xb)
                vl     += crit(pred, yb).item()
                preds.extend(pred.cpu().numpy())
                trues.extend(yb.numpy())

        vl   /= len(te_ld)
        vauc  = roc_auc_score(trues, preds)
        sched.step(vl)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_auc"].append(vauc)

        if vauc > best_auc:
            best_auc   = vauc
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"{epoch:>6} {tl:>12.4f} {vl:>10.4f} {vauc:>8.4f}")

    model.load_state_dict(best_state)
    print(f"\n  Best Transfer AUC: {best_auc:.4f}")

    # Final metrics
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            preds.extend(model(xb.to(DEVICE)).cpu().numpy())
            trues.extend(yb.numpy())

    pb = (np.array(preds) >= 0.5).astype(int)
    tf_metrics = {
        "auc":       roc_auc_score(trues, preds),
        "precision": precision_score(trues, pb, zero_division=0),
        "recall":    recall_score(trues, pb, zero_division=0),
        "f1":        f1_score(trues, pb, zero_division=0),
    }
    print("\n  Transfer Function Metrics:")
    for k, v in tf_metrics.items():
        print(f"    {k:<12}: {v:.4f}")

    return model, history, tf_metrics, np.array(preds), np.array(trues)


# ══════════════════════════════════════════════════════════════
# SECTION 8 — PHASE 4: RECOMMEND FROM OPTISCHOLAR
# ══════════════════════════════════════════════════════════════

def get_proxy_embedding(profile, student_embs, le_deg, le_ses,
                         le_gender, scaler):
    """
    New student not in Nigerian dataset.
    Find proxy embedding by matching profile to nearest
    GPA/SES/degree percentile band of Nigerian students.
    """
    gpa    = float(profile.get("gpa_proxy") or
                   profile.get("final_gpa") or 2.5)
    income = float(profile.get("household_income", 50000))
    age    = float(profile.get("age", 20))

    cont   = scaler.transform([[gpa, income, age]])[0]
    gpa_n  = cont[0]

    # Segment students into GPA thirds and return mean embedding
    n = len(student_embs)
    if gpa_n >= 0.66:
        seg = student_embs[int(n*0.66):]
    elif gpa_n >= 0.33:
        seg = student_embs[int(n*0.33):int(n*0.66)]
    else:
        seg = student_embs[:int(n*0.33)]

    return seg.mean(axis=0)


def recommend_optischolar(profile, transfer_fn, opto_df,
                           student_embs, le_type,
                           le_deg, le_ses, le_gender, scaler,
                           top_n=10, verbose=True):
    """
    Phase 4: Score all eligible OptiScholar scholarships using
    the trained transfer function.
    """
    stu_emb = get_proxy_embedding(
        profile, student_embs,
        le_deg, le_ses, le_gender, scaler
    )

    level     = profile.get("degree_level",
                profile.get("study_level", "bachelor"))
    level_col = f"eligible_{level}"
    if level_col not in opto_df.columns:
        level_col = "eligible_bachelor"

    gpa  = float(profile.get("gpa_proxy") or
                 profile.get("final_gpa") or 0)
    intl = int(profile.get("International", 0))
    need = int(profile.get("financial_need", 0))

    # Hard eligibility filter
    cands = opto_df[opto_df[level_col] == 1].copy()
    cands = cands[
        (cands["min_gpa_required"] <= 0) |
        (cands["min_gpa_required"] <= gpa * 5.0)
    ]
    if intl:
        cands = cands[
            ~cands["citizenship_required"].str.contains(
                "us_citizen|specific_residency",
                case=False, na=False
            )
        ]

    if cands.empty:
        print("  No candidates after eligibility filter.")
        return pd.DataFrame()

    # Build [student_emb(32) + sch_features(5)] for each candidate
    amt_max = opto_df["funding_amount_raw"].max()

    def safe_type_enc(t):
        t = str(t).strip()
        return le_type.transform([t])[0] if t in le_type.classes_ else 0

    X_list = []
    for _, row in cands.iterrows():
        type_enc = safe_type_enc(row.get("scholarship_type", "Merit-Based"))
        amount_n = float(row.get("funding_amount_raw", 0)) / (amt_max + 1e-8)
        gpa_n    = scaler.transform([[gpa, 50000, 20]])[0][0]
        inc_n    = scaler.transform([[2.5,
                   float(profile.get("household_income", 50000)),
                   20]])[0][1]
        deg_enc  = 1 if level == "bachelor" else 0

        sch_vec  = np.array([type_enc, amount_n, gpa_n, inc_n, deg_enc],
                             dtype=float)
        X_list.append(np.concatenate([stu_emb, sch_vec]))

    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(DEVICE)

    transfer_fn.eval()
    with torch.no_grad():
        scores = transfer_fn(X).cpu().numpy()

    cands = cands.copy().reset_index(drop=True)
    # Normalize scores to [0,1] range to prevent sigmoid saturation display issue
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-6:
        cands["ncf_score"] = (scores - s_min) / (s_max - s_min)
    else:
        cands["ncf_score"] = scores
    result = cands.sort_values("ncf_score", ascending=False).head(top_n)


    if verbose:
        print(f"\n{'='*65}")
        print(f"  NCF Transfer Recommendations")
        print(f"  Student  : {profile.get('student_id','UNKNOWN')}")
        print(f"  Eligible : {len(cands):,} candidates")
        print(f"{'='*65}")
        for _, r in result.iterrows():
            print(f"  {r['scholarship_id']:<10} "
                  f"{str(r['scholarship_title'])[:45]:<45} "
                  f"score={r['ncf_score']:.3f}  "
                  f"[{r.get('scholarship_type','')}]")

    return result

# ══════════════════════════════════════════════════════════════
# SECTION 9 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def visualize_ncf_training(ncf_history, ncf_metrics,
                            ncf_preds, ncf_trues):
    print("\n[Visualizing NCF Training]")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("NCF — Training & Evaluation",
                 fontsize=13, fontweight="bold")

    # Loss curves
    epochs = range(1, len(ncf_history["train_loss"]) + 1)
    axes[0].plot(epochs, ncf_history["train_loss"],
                 color=COLOR_POS, label="Train Loss", lw=2)
    axes[0].plot(epochs, ncf_history["val_loss"],
                 color=COLOR_NEG, label="Val Loss", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # AUC curve
    axes[1].plot(epochs, ncf_history["val_auc"],
                 color=COLOR_ACC, lw=2)
    axes[1].axhline(y=max(ncf_history["val_auc"]),
                    color="red", linestyle="--", alpha=0.6,
                    label=f"Best AUC={max(ncf_history['val_auc']):.3f}")
    axes[1].axhline(y=0.5, color="black", linestyle=":",
                    alpha=0.4, label="Random baseline")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Validation AUC Over Training")
    axes[1].legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(ncf_trues, ncf_preds)
    roc_auc     = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color=COLOR_POS, lw=2,
                 label=f"NCF AUC={roc_auc:.3f}")
    axes[2].plot([0,1],[0,1], "k--", lw=1, label="Random")
    axes[2].fill_between(fpr, tpr, alpha=0.1, color=COLOR_POS)
    axes[2].set_xlabel("FPR")
    axes[2].set_ylabel("TPR")
    axes[2].set_title("ROC Curve")
    axes[2].legend()

    plt.tight_layout()
    _save(fig, "01_ncf_training")


def visualize_transfer_training(tf_history, tf_metrics,
                                 tf_preds, tf_trues):
    print("[Visualizing Transfer Function Training]")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Transfer Function — Training & Evaluation",
                 fontsize=13, fontweight="bold")

    epochs = range(1, len(tf_history["train_loss"]) + 1)
    axes[0].plot(epochs, tf_history["train_loss"],
                 color=COLOR_POS, label="Train Loss", lw=2)
    axes[0].plot(epochs, tf_history["val_loss"],
                 color=COLOR_NEG, label="Val Loss", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, tf_history["val_auc"],
                 color=COLOR_ACC, lw=2)
    axes[1].axhline(y=max(tf_history["val_auc"]),
                    color="red", linestyle="--", alpha=0.6,
                    label=f"Best AUC={max(tf_history['val_auc']):.3f}")
    axes[1].axhline(y=0.5, color="black", linestyle=":",
                    alpha=0.4, label="Random baseline")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Validation AUC Over Training")
    axes[1].legend()

    fpr, tpr, _ = roc_curve(tf_trues, tf_preds)
    roc_auc     = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color=COLOR_ACC, lw=2,
                 label=f"Transfer AUC={roc_auc:.3f}")
    axes[2].plot([0,1],[0,1], "k--", lw=1)
    axes[2].fill_between(fpr, tpr, alpha=0.1, color=COLOR_ACC)
    axes[2].set_xlabel("FPR")
    axes[2].set_ylabel("TPR")
    axes[2].set_title("ROC Curve")
    axes[2].legend()

    plt.tight_layout()
    _save(fig, "02_transfer_training")


def visualize_embeddings(student_embs, merged):
    print("[Visualizing Student Embeddings]")

    from sklearn.manifold import TSNE

    # Sample 800 students for t-SNE
    n_sample = min(800, len(student_embs))
    idx      = np.random.choice(len(student_embs),
                                 n_sample, replace=False)
    emb_s    = student_embs[idx]

    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(emb_s)

    # Get GPA and approval rate for sampled students
    # Use student index to get degree level
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("NCF Student Embedding Space (t-SNE)",
                 fontsize=13, fontweight="bold")

    # Colour by embedding magnitude (proxy for "scholarship-worthiness")
    magnitudes = np.linalg.norm(emb_s, axis=1)
    sc1 = axes[0].scatter(emb_2d[:,0], emb_2d[:,1],
                           c=magnitudes, cmap="viridis",
                           alpha=0.6, s=15)
    plt.colorbar(sc1, ax=axes[0], label="Embedding Magnitude")
    axes[0].set_title("Coloured by Embedding Magnitude")
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")

    # Colour by embedding dim 0 (most discriminative)
    sc2 = axes[1].scatter(emb_2d[:,0], emb_2d[:,1],
                           c=emb_s[:,0], cmap="coolwarm",
                           alpha=0.6, s=15)
    plt.colorbar(sc2, ax=axes[1], label="Embedding dim 0")
    axes[1].set_title("Coloured by Embedding Dimension 0")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    _save(fig, "03_student_embeddings_tsne")


def visualize_recommendations(results_dict, opto_df):
    print("[Visualizing Recommendations]")

    fig, axes = plt.subplots(1, len(results_dict),
                              figsize=(13 * len(results_dict), 6))
    if len(results_dict) == 1:
        axes = [axes]

    fig.suptitle("NCF Transfer — Top 10 Recommendations per Profile",
                 fontsize=13, fontweight="bold")

    for ax, (label, result) in zip(axes, results_dict.items()):
        if result.empty:
            ax.text(0.5, 0.5, "No results", ha="center", va="center")
            continue
        titles = [t[:40]+"..." if len(t)>40 else t
                  for t in result["scholarship_title"].tolist()]
        scores = result["ncf_score"].values
        x_min  = scores.min() * 0.97
        x_max  = scores.max() * 1.02
        bars   = ax.barh(titles[::-1], scores[::-1], color=COLOR_4)
        ax.set_xlim(x_min, x_max)
        for bar, val in zip(bars, scores[::-1]):
            ax.text(x_min + (x_max-x_min)*0.98,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="right",
                    fontsize=8, color="white", fontweight="bold")
        ax.set_xlabel("NCF Score")
        ax.set_title(f"Profile: {label}", fontsize=10)

    plt.tight_layout()
    _save(fig, "04_ncf_recommendations")


def visualize_comparison(ncf_metrics, tf_metrics):
    print("[Visualizing Model Comparison]")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("NCF Pipeline — Model Comparison",
                 fontsize=13, fontweight="bold")

    # AUC comparison
    models = ["NCF\n(Nigerian)", "Transfer Fn\n(Nigerian→OptiScholar)"]
    aucs   = [ncf_metrics["auc"], tf_metrics["auc"]]
    bars   = axes[0].bar(models, [a*100 for a in aucs],
                          color=[COLOR_POS, COLOR_ACC])
    axes[0].set_ylim(40, 100)
    axes[0].set_ylabel("ROC-AUC (%)")
    axes[0].set_title("AUC Comparison")
    axes[0].axhline(y=50, color="black", linestyle="--",
                    alpha=0.4, label="Random baseline")
    for bar, val in zip(bars, aucs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     val*100 + 0.5, f"{val*100:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")

    # Precision / Recall / F1
    met_labels = ["Precision", "Recall", "F1"]
    ncf_vals   = [ncf_metrics["precision"],
                  ncf_metrics["recall"],
                  ncf_metrics["f1"]]
    tf_vals    = [tf_metrics["precision"],
                  tf_metrics["recall"],
                  tf_metrics["f1"]]
    x = np.arange(3)
    w = 0.35
    axes[1].bar(x - w/2, ncf_vals, w, label="NCF",         color=COLOR_POS)
    axes[1].bar(x + w/2, tf_vals,  w, label="Transfer Fn", color=COLOR_ACC)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(met_labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Precision / Recall / F1")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, "05_ncf_comparison")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Load
    merged, students, opto_df = load_data()

    # Encoders
    (le_stu, le_sch, le_type,
     le_deg, le_ses, le_gender, scaler) = build_encoders(merged, students)

    # Encode features
    data_enc = encode_features(
        merged, le_stu, le_sch, le_type,
        le_deg, le_ses, le_gender, scaler
    )

    # Negative sampling
    data_aug = negative_sample(data_enc, n_neg=1)  # 1:1 ratio reduces overfitting

    # Re-encode after augmentation (new rows need encoding)
    data_aug = encode_features(
        data_aug, le_stu, le_sch, le_type,
        le_deg, le_ses, le_gender, scaler
    )

    # Phase 1: Train NCF
    ncf_model, ncf_hist, ncf_metrics, ncf_preds, ncf_trues = train_ncf(
        data_aug, le_stu, le_sch,
        epochs=100, batch_size=512,
        emb_dim=8, hidden=[32, 16], dropout=0.4
    )

    # Phase 2: Extract embeddings
    student_embs = extract_embeddings(ncf_model, le_stu)

    # Phase 3: Train transfer function
    (transfer_fn, tf_hist,
     tf_metrics, tf_preds, tf_trues) = train_transfer_function(
        data_enc, student_embs, le_stu, le_type,
        epochs=80, batch_size=512
    )

    # Visualize training
    visualize_ncf_training(ncf_hist, ncf_metrics, ncf_preds, ncf_trues)
    visualize_transfer_training(tf_hist, tf_metrics, tf_preds, tf_trues)
    visualize_embeddings(student_embs, merged)
    visualize_comparison(ncf_metrics, tf_metrics)

    # Phase 4: Recommend from OptiScholar
    profiles = [
        {
            "student_id":      "P001_HighGPA",
            "degree_level":    "bachelor",
            "study_level":     "bachelor",
            "gpa_proxy":       3.7,
            "final_gpa":       3.7,
            "financial_need":  0,
            "International":   0,
            "age":             21,
            "gender":          "Male",
            "ses_category":    "High",
            "household_income": 85000,
        },
        {
            "student_id":      "P002_LowIncome",
            "degree_level":    "bachelor",
            "study_level":     "bachelor",
            "gpa_proxy":       2.8,
            "final_gpa":       2.8,
            "financial_need":  1,
            "International":   0,
            "age":             20,
            "gender":          "Female",
            "ses_category":    "Low",
            "household_income": 12000,
        },
    ]

    results = {}
    for p in profiles:
        res = recommend_optischolar(
            p, transfer_fn, opto_df, student_embs,
            le_type, le_deg, le_ses, le_gender, scaler,
            top_n=10
        )
        results[p["student_id"]] = res

    visualize_recommendations(results, opto_df)

    # Save models
    torch.save(ncf_model.state_dict(),   "ncf_model_v2.pt")
    torch.save(transfer_fn.state_dict(), "transfer_fn_v2.pt")
    print("\nSaved: ncf_model_v2.pt, transfer_fn_v2.pt")
    print(f"Figures saved to: ./{OUT_DIR}/")
