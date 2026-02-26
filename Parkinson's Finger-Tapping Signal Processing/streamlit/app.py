import streamlit as st
import numpy as np
import joblib
from scipy.signal import savgol_filter, find_peaks

st.title("Parkinson’s Finger Tapping Classifier")

# =========================
# Load Models
# =========================

LEFT_MODEL_PATH = "models/svm_left.pkl"
RIGHT_MODEL_PATH = "models/svm_right.pkl"

left_model = joblib.load(LEFT_MODEL_PATH)
right_model = joblib.load(RIGHT_MODEL_PATH)

# =========================
# Sidebar Inputs
# =========================

hand = st.sidebar.selectbox("Select Hand", ["Left", "Right"])

updrs_raw = st.sidebar.selectbox(
    "UPDRS Finger Tapping Score",
    options=[0,1,2,3,4]
)

st.sidebar.write("Upload or paste signals")

amplitude_input = st.text_area(
    "Amplitude values (comma separated)",
    height=150
)

time_input = st.text_area(
    "Time values (comma separated)",
    height=150
)

# =========================
# Helpers
# =========================

TARGET_LEN = 1800

def parse_signal(text):
    return np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""])

def pad_signal(amplitude, time):
    if len(amplitude) < TARGET_LEN:
        amplitude = np.pad(amplitude, (0, TARGET_LEN-len(amplitude)), mode='edge')
    if len(time) < TARGET_LEN:
        time = np.pad(time, (0, TARGET_LEN-len(time)), constant_values=30.0)
    return amplitude, time

def preprocess_signal(amplitude):
    window = 7 if len(amplitude) > 7 else len(amplitude)-1
    if window % 2 == 0:
        window += 1
    return savgol_filter(amplitude, window_length=window, polyorder=2)

def extract_features(amplitude, time):
    amp = preprocess_signal(amplitude)

    peaks, _ = find_peaks(
        amp,
        prominence=0.1 * (amp.max() - amp.min()),
        distance=10
    )

    if len(peaks) < 5:
        return None

    peak_amps = amp[peaks]

    dt = np.median(np.diff(time)[np.diff(time) > 0])
    itis = np.diff(peaks) * dt

    early = peak_amps[:max(3, int(0.2*len(peak_amps)))]
    late  = peak_amps[-max(3, int(0.2*len(peak_amps))):]

    features = {
        "num_taps": len(peaks),
        "mean_peak_amp": peak_amps.mean(),
        "std_peak_amp": peak_amps.std(),
        "amp_decrement": early.mean() - late.mean(),
        "mean_iti": itis.mean(),
        "std_iti": itis.std(),
        "cv_iti": itis.std()/itis.mean(),
        "num_long_pauses": np.sum(itis > 1.5*np.median(itis)),
        "prop_long_pauses": np.mean(itis > 1.5*np.median(itis))
    }

    return features

# =========================
# Prediction
# =========================

if st.button("Predict"):

    if amplitude_input == "" or time_input == "":
        st.error("Please provide amplitude and time values.")
    else:
        amplitude = parse_signal(amplitude_input)
        time = parse_signal(time_input)

        amplitude, time = pad_signal(amplitude, time)

        features = extract_features(amplitude, time)

        if features is None:
            st.error("Not enough valid taps detected.")
        else:
            # encode UPDRS
            updrs_encoded = 1 if updrs_raw >= 3 else 0
            features["updrs"] = updrs_encoded

            feature_vector = np.array(list(features.values())).reshape(1, -1)

            model = left_model if hand == "Left" else right_model

            prob = model.predict_proba(feature_vector)[0][1]

            if prob < 0.40:
                prediction = "Negative (No Parkinson’s)"
            elif prob > 0.55:
                prediction = "Positive (Parkinson’s)"
            else:
                prediction = "Uncertain, further evaluation needed"

            st.subheader("Prediction")
            st.write(prediction)
            st.write(f"Probability: {prob:.3f}")
            st.write("Encoded UPDRS:", updrs_encoded)