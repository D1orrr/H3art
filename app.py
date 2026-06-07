"""
Heart Disease Risk Screening Tool
Streamlit web application powered by XGBoost trained on BRFSS 2023 data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="H3art - Cardiovascular Risk Screener",
    layout="wide",
)

# Hide Streamlit's default chrome + running-man status indicator
st.markdown("""
    <style>
        /* Hide Streamlit hamburger menu, footer, deploy button */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none !important;}

        /* Hide the "running man" status widget at top-right */
        [data-testid="stStatusWidget"] {display: none !important;}

        /* Hide the entire toolbar (deploy + status + menu) as a belt-and-braces */
        [data-testid="stToolbar"] {visibility: hidden !important;}

        /* Hide top decoration bar */
        [data-testid="stDecoration"] {display: none !important;}

        /* Force the default Streamlit spinner to be a clean circular ring */
        .stSpinner > div {
            border-width: 4px !important;
            border-top-color: #2E75B6 !important;
            border-right-color: rgba(46, 117, 182, 0.25) !important;
            border-bottom-color: rgba(46, 117, 182, 0.25) !important;
            border-left-color: rgba(46, 117, 182, 0.25) !important;
        }

        /* Hide the anchor link icons that appear on hover next to headers/titles */
        [data-testid="stHeaderActionElements"] {display: none !important;}
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {display: none !important;}
        .stMarkdown h1 > div > a,
        .stMarkdown h2 > div > a,
        .stMarkdown h3 > div > a,
        .stMarkdown h4 > div > a,
        .stMarkdown h5 > div > a,
        .stMarkdown h6 > div > a {display: none !important;}

        /* Bold + custom font for the primary Predict Risk button */
        .stButton > button[kind="primary"] {
            font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif !important;
            font-weight: 1200 !important;
            font-size: 1.15rem !important;
            letter-spacing: 0.6px !important;
            text-transform: uppercase !important;
            padding-top: 0.65rem !important;
            padding-bottom: 0.65rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    with open('xgboost_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Sentinel for "Unknown" -- becomes NaN, which XGBoost handles natively
UNK = None

# ============================================================
# Helpers
# ============================================================
def age_to_bracket(age_years: int) -> int:
    """Map a numeric age to BRFSS _AGEG5YR bracket (1-13)."""
    if age_years < 18: return 1
    if age_years <= 24: return 1
    if age_years <= 29: return 2
    if age_years <= 34: return 3
    if age_years <= 39: return 4
    if age_years <= 44: return 5
    if age_years <= 49: return 6
    if age_years <= 54: return 7
    if age_years <= 59: return 8
    if age_years <= 64: return 9
    if age_years <= 69: return 10
    if age_years <= 74: return 11
    if age_years <= 79: return 12
    return 13  # 80+

def compute_bmi_category(weight_kg: float, height_m: float):
    """Return (BMI value, category code 1-4, category label)."""
    if weight_kg <= 0 or height_m <= 0:
        return None, UNK, ""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:    return bmi, 1, f"Underweight (BMI {bmi:.1f})"
    if bmi < 25.0:    return bmi, 2, f"Normal weight (BMI {bmi:.1f})"
    if bmi < 30.0:    return bmi, 3, f"Overweight (BMI {bmi:.1f})"
    return bmi, 4, f"Obese (BMI {bmi:.1f})"

def classify_bp(sys: int, dia: int):
    """Return (category label, severity, model code 0/1) using AHA guidelines."""
    if sys > 180 or dia > 120:
        return "Hypertensive Crisis — seek immediate medical care", "critical", 1
    if sys >= 140 or dia >= 90:
        return "Stage 2 Hypertension", "high", 1
    if sys >= 130 or dia >= 80:
        return "Stage 1 Hypertension", "moderate", 1
    if sys >= 120:
        return "Elevated blood pressure", "low", 0
    return "Normal blood pressure", "normal", 0

def classify_chol(total: int):
    """Return (label, severity, model code 0/1)."""
    if total >= 240:
        return "High cholesterol", "high", 1
    if total >= 200:
        return "Borderline-high cholesterol", "moderate", 1
    return "Desirable cholesterol", "normal", 0

def classify_glucose(glucose: int):
    """Return (label, severity, model code 0/1/2) from fasting glucose mg/dL."""
    if glucose >= 126:
        return "Diabetes range", "high", 2
    if glucose >= 100:
        return "Pre-diabetes range", "moderate", 1
    return "Normal blood sugar", "normal", 0

def severity_badge(label: str, severity: str):
    """Render a colored info/warning/error badge."""
    if severity == "normal":  st.success(label)
    elif severity == "low":   st.info(label)
    elif severity == "moderate": st.warning(label)
    else:                     st.error(label)

# ============================================================
# Header
# ============================================================
st.title("H3art - Cardiovascular Risk Screener")
st.markdown(
    "##### *Your cardiovascular risk screening companion*"
)
st.markdown(
    "Enter your health information below to estimate your cardiovascular risk. "
    "Select **Unknown** for anything you are not sure about — the model still "
    "produces an estimate from what you do provide. Powered by an XGBoost model "
    "trained on the CDC BRFSS 2023 dataset."
)
st.divider()

# ============================================================
# Inputs
# ============================================================
col1, col2 = st.columns(2, gap="large")

# ============================================================
# LEFT COLUMN: Demographics, Body, Lifestyle
# ============================================================
with col1:
    st.subheader("Demographics & Lifestyle")

    # ---- Age (numeric input) ----
    age_years = st.number_input(
        "Age (years)",
        min_value=18, max_value=120, value=40, step=1,
    )
    age = age_to_bracket(age_years)

    # ---- Sex ----
    sex = st.selectbox(
        "Sex",
        options=[("Female", 0), ("Male", 1)],
        format_func=lambda x: x[0],
    )[1]

    # ---- Height + Weight -> BMI ----
    st.markdown("**Height & Weight**")
    unit_system = st.radio(
        "Units", options=["Metric (cm / kg)", "Imperial (ft-in / lbs)"],
        horizontal=True, label_visibility="collapsed",
    )

    bmi_unknown = st.checkbox("I don't know my height/weight")
    if bmi_unknown:
        bmi_code = UNK
        bmi_val = None
    else:
        if unit_system.startswith("Metric"):
            hc1, hc2 = st.columns(2)
            with hc1:
                height_cm = st.number_input("Height (cm)", 100, 250, 170)
            with hc2:
                weight_kg = st.number_input("Weight (kg)", 30, 300, 70)
            height_m = height_cm / 100.0
        else:
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                feet = st.number_input("Height (ft)", 3, 8, 5)
            with hc2:
                inches = st.number_input("Inches", 0, 11, 7)
            with hc3:
                weight_lbs = st.number_input("Weight (lbs)", 60, 600, 155)
            height_m = (feet * 12 + inches) * 0.0254
            weight_kg = weight_lbs * 0.453592

        bmi_val, bmi_code, bmi_label = compute_bmi_category(weight_kg, height_m)
        if bmi_label:
            severity_str = "normal" if bmi_code == 2 else ("moderate" if bmi_code == 3 else ("low" if bmi_code == 1 else "high"))
            severity_badge(bmi_label, severity_str)

    # ---- Smoking (no more "100 cigarettes" phrasing) ----
    smoking_selection = st.selectbox(
        "Smoking status",
        options=[
            ("Never smoked",                              0,   "never"),
            ("Former smoker (used to smoke but quit)",   1,   "former"),
            ("Current smoker",                            1,   "current"),
            ("Unknown",                                   UNK, "unknown"),
        ],
        format_func=lambda x: x[0],
    )
    smoking = smoking_selection[1]

    # Conditional follow-up for current smokers
    cigs_per_day = None
    if smoking_selection[2] == "current":
        cigs_per_day = st.number_input(
            "How many cigarettes per day, on average?",
            min_value=1, max_value=100, value=10, step=1,
            help="This figure provides personal context but does not directly "
                 "change the model's prediction — the model uses smoking as a "
                 "Yes/No indicator from the BRFSS dataset.",
        )
        if cigs_per_day <= 10:
            st.info(
                f"Light smoking ({cigs_per_day}/day). Even low-volume "
                f"smoking measurably raises cardiovascular risk."
            )
        elif cigs_per_day <= 20:
            st.warning(
                f"Moderate smoking ({cigs_per_day}/day, ≈ 1 pack). "
                f"Significantly elevated CVD risk. Quitting reduces this risk substantially."
            )
        else:
            st.error(
                f"Heavy smoking ({cigs_per_day}/day, ≈ {cigs_per_day/20:.1f} packs). "
                f"Strongly linked to elevated CVD risk. Please consider speaking "
                f"with a healthcare professional about cessation support."
            )

    # ---- Alcohol (category dropdown) ----
    alcohol_choice = st.selectbox(
        "Alcohol consumption (drinks per week)",
        options=[
            ("I don't drink",              "none"),
            ("Light: 1 – 3 drinks/week",   "light"),
            ("Moderate: 4 – 7 drinks/week","moderate"),
            ("8 – 14 drinks/week",         "high"),
            ("15+ drinks/week",            "very_high"),
            ("Unknown",                    "unknown"),
        ],
        format_func=lambda x: x[0],
    )[1]

    # Map to "Heavy Alcohol" binary using sex-aware thresholds
    # (Men > 14/week OR Women > 7/week per BRFSS _RFDRHV8 definition)
    if alcohol_choice == "unknown":
        alcohol = UNK
    elif alcohol_choice == "very_high":
        alcohol = 1
    elif alcohol_choice == "high":     # 8-14: heavy only if female (>7)
        alcohol = 1 if sex == 0 else 0
    else:                              # 0, light, moderate (1-7)
        alcohol = 0

    # ---- Exercise frequency ----
    exercise_freq = st.selectbox(
        "How often do you exercise?",
        options=[
            ("No exercise at all", 0.0),
            ("Less than once a week", 2.0),
            ("1 – 2 times per week", 6.5),
            ("3 – 4 times per week", 15.0),
            ("5 – 7 times per week (daily)", 26.0),
            ("Unknown", UNK),
        ],
        format_func=lambda x: x[0],
    )[1]

    # ---- Physical health days (replaces General Health) ----
    ph_unknown = st.checkbox("I don't know / prefer not to say (physical health)")
    if ph_unknown:
        physical_health = UNK
    else:
        physical_health = st.slider(
            "In the past 30 days, how many days was your physical health not good? "
            "(includes illness, injury, fatigue)",
            min_value=0, max_value=30, value=0,
        )

    # ---- Mental health days ----
    mh_unknown = st.checkbox("I don't know / prefer not to say (mental health)")
    if mh_unknown:
        mental_health = UNK
    else:
        mental_health = st.slider(
            "In the past 30 days, how many days was your mental health not good?",
            min_value=0, max_value=30, value=0,
        )

# ============================================================
# RIGHT COLUMN: Medical history
# ============================================================
with col2:
    st.subheader("Medical History")
    st.caption("Have you ever been told by a doctor or health professional that you have…")

    # ---------- BLOOD PRESSURE (hybrid input) ----------
    st.markdown("**Blood Pressure**")
    bp_method = st.selectbox(
        "How would you like to answer?",
        options=["No", "Yes — I've been diagnosed with high blood pressure",
                 "Check for specifics", "Unknown"],
        key="bp_method", label_visibility="collapsed",
    )
    if bp_method == "No":
        high_bp = 0
    elif bp_method.startswith("Yes"):
        high_bp = 1
    elif bp_method == "Unknown":
        high_bp = UNK
    else:
        bpc1, bpc2 = st.columns(2)
        with bpc1:
            systolic = st.number_input("Systolic (top number, mmHg)", 70, 250, 120)
        with bpc2:
            diastolic = st.number_input("Diastolic (bottom, mmHg)", 40, 150, 80)
        bp_label, bp_severity, bp_code = classify_bp(systolic, diastolic)
        severity_badge(bp_label, bp_severity)
        high_bp = bp_code

    # ---------- CHOLESTEROL (hybrid input) ----------
    st.markdown("**Cholesterol**")
    chol_method = st.selectbox(
        "How would you like to answer?",
        options=["No", "Yes — I've been diagnosed with high cholesterol",
                 "Check for specifics", "Unknown"],
        key="chol_method", label_visibility="collapsed",
    )
    if chol_method == "No":
        high_chol = 0
    elif chol_method.startswith("Yes"):
        high_chol = 1
    elif chol_method == "Unknown":
        high_chol = UNK
    else:
        total_chol = st.number_input("Total cholesterol (mg/dL)", 80, 500, 180)
        chol_label, chol_severity, chol_code = classify_chol(total_chol)
        severity_badge(chol_label, chol_severity)
        high_chol = chol_code

    # ---------- DIABETES / BLOOD SUGAR (hybrid input) ----------
    st.markdown("**Diabetes / Blood Sugar**")
    diab_method = st.selectbox(
        "How would you like to answer?",
        options=["No", "Pre-diabetes / borderline",
                 "Yes — I've been diagnosed with diabetes",
                 "Check for specifics (Fasting is required)", "Unknown"],
        key="diab_method", label_visibility="collapsed",
    )
    if diab_method == "No":
        diabetes = 0
    elif diab_method.startswith("Pre"):
        diabetes = 1
    elif diab_method.startswith("Yes"):
        diabetes = 2
    elif diab_method == "Unknown":
        diabetes = UNK
    else:
        glucose = st.number_input("Fasting blood glucose (mg/dL)", 50, 500, 90)
        gl_label, gl_severity, gl_code = classify_glucose(glucose)
        severity_badge(gl_label, gl_severity)
        diabetes = gl_code

    st.divider()

    # ---------- Remaining yes/no conditions ----------
    yes_no_unk = [("No", 0), ("Yes", 1), ("Unknown", UNK)]

    stroke = st.selectbox(
        "Have you ever had a stroke?",
        options=yes_no_unk, format_func=lambda x: x[0],
    )[1]

    other_cancer = st.selectbox(
        "Have you ever had cancer",
        options=yes_no_unk, format_func=lambda x: x[0],
    )[1]

    copd = st.selectbox(
        "COPD, emphysema, or chronic bronchitis",
        options=yes_no_unk, format_func=lambda x: x[0],
    )[1]

    kidney = st.selectbox(
        "Kidney disease",
        options=yes_no_unk, format_func=lambda x: x[0],
        help="Does not include kidney stones, bladder infection, or incontinence.",
    )[1]
    kidney_stones = st.checkbox(
        "I also have a history of kidney stones",
        help="Kidney stones are tracked separately from kidney disease. "
             "Having stones does not directly raise heart-disease risk, but "
             "they share lifestyle risk factors (dehydration, diet). Consult "
             "your doctor for stone-specific advice."
    )

    depression = st.selectbox(
        "Depressive disorder (depression, major depression, dysthymia)",
        options=yes_no_unk, format_func=lambda x: x[0],
    )[1]

# ============================================================
# Prediction
# ============================================================
st.divider()
predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing your cardiovascular risk..."):
        user_data = pd.DataFrame(
            {
                'Age':            [age],
                'Sex':            [sex],
                'BMICategory':    [bmi_code],
                'Smoking':        [smoking],
                'Alcohol':        [alcohol],
                'ExerciseFreq':   [exercise_freq],
                'HighBP':         [high_bp],
                'HighChol':       [high_chol],
                'Diabetes':       [diabetes],
                'Stroke':         [stroke],
                'OtherCancer':    [other_cancer],
                'COPD':           [copd],
                'KidneyDisease':  [kidney],
                'Depression':     [depression],
                'MentalHealth':   [mental_health],
                'PhysicalHealth': [physical_health],
            },
            dtype=float,
        )

        prediction  = int(model.predict(user_data)[0])
        probability = float(model.predict_proba(user_data)[0][1]) * 100

    if prediction == 1:
        st.error(f"### Elevated Risk — {probability:.1f}% probability")
        st.markdown(
            "The model indicates an **elevated probability** of cardiovascular "
            "risk. Please consult a healthcare professional for proper evaluation. "
            "Early intervention can significantly reduce long-term risk."
        )
    else:
        st.success(f"### Low Risk — {probability:.1f}% probability")
        st.markdown(
            "The model indicates a **low probability** of cardiovascular risk. "
            "Continue with healthy lifestyle habits and regular check-ups."
        )

    st.progress(min(probability / 100, 1.0))

    if kidney_stones:
        st.info(
            "You indicated a history of kidney stones. While stones are not a "
            "direct cardiovascular risk factor, they often share underlying "
            "lifestyle factors (hydration, diet, blood pressure). Mention this "
            "to your physician during your next check-up."
        )

    st.caption(
        "**Disclaimer**: This tool provides a statistical risk estimate based "
        "on self-reported survey data from the CDC BRFSS 2023 study, and is "
        "**not** a medical diagnosis. Always consult a qualified healthcare "
        "provider for medical decisions."
    )