import streamlit as st
import numpy as np
import soundfile as sf
from sklearn.neighbors import KNeighborsClassifier

st.title("AI-Based Noise Level Classifier")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    # Play audio
    st.subheader("🔊 Listen to Uploaded Audio")
    st.audio(uploaded_file, format="audio/wav")

    # Read audio
    audio, sample_rate = sf.read(uploaded_file)

    if audio.ndim > 1:
        audio = audio[:, 0]

    # Feature extraction
    rms = np.sqrt(np.mean(audio**2))
    loudness = rms * 1000

    st.write("📈 Loudness Value:", round(loudness, 2))

    # Training data
    X = np.array([[10], [20], [35], [50], [65], [80], [95]])
    y = np.array([0, 0, 0, 1, 1, 2, 2])

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    prediction = model.predict([[loudness]])
    labels = ["LOW", "MEDIUM", "HIGH"]
    level = labels[prediction[0]]

    st.subheader("🔍 Noise Level:")
    st.success(level)

    # -------- Messages & Actions --------
    if level == "LOW":
        st.info("😊 Peaceful environment detected!")
        st.write("""
        ✔ Suitable for studying, sleeping, meditation  
        ✔ No health risk  
        ✔ Ideal indoor noise level  
        """)
        st.balloons()

    elif level == "MEDIUM":
        st.warning("🙂 Moderate noise level.")
        st.write("""
        ⚠ May cause distraction over long periods  
        ✔ Acceptable for offices and classrooms  
        ✔ Consider lowering volume if possible  
        """)

    else:  # HIGH
        st.error("🚨 High noise detected!")
        st.write("""
        ❗ Risk of stress and hearing discomfort  
        ❗ Avoid long exposure  
        ✔ Use ear protection  
        ✔ Reduce noise source immediately  
        """)

