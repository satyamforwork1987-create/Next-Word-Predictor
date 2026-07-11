import streamlit as st
import numpy as np
import pickle
import time
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Next Word AI", page_icon="🤖", layout="centered")

# -----------------------------
# 🌈 FORCE REMOVE BLACK BACKGROUND
# -----------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND FIX */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ff00cc, #333399, #00c6ff);
    background-attachment: fixed;
}

/* REMOVE HEADER BG */
[data-testid="stHeader"] {
    background: transparent;
}

/* MAIN AREA */
[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

/* GLASS UI */
.block-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(15px);
}

/* TITLE */
h1 {
    text-align: center;
    color: white;
    text-shadow: 0px 0px 20px #00ffff;
}

/* INPUT */
.stTextInput input {
    border-radius: 12px;
    padding: 12px;
    background-color: rgba(255,255,255,0.9);
    color: black;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #ff00cc, #00c6ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    transition: 0.3s;
}

/* HOVER */
.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0px 0px 25px #ff00cc, 0px 0px 25px #00c6ff;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🌈 HERO ANIMATION BOX
# -----------------------------
particles_js = """
<!DOCTYPE html>
<html>
<head>
<style>
#particles-box {
  width: 80%;
  height: 350px;
  margin: auto;
  border-radius: 20px;
  overflow: hidden;
  position: relative;
  background: linear-gradient(135deg, #ff00cc, #333399, #00c6ff);
  background-size: 400% 400%;
  animation: gradientBG 10s ease infinite;
}

@keyframes gradientBG {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

.overlay-text {
  position: absolute;
  top: 45%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: white;
  font-size: 30px;
  font-weight: bold;
  text-shadow: 0px 0px 20px #00ffff;
}

.sub-text {
  position: absolute;
  top: 60%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #eeeeee;
  font-size: 16px;
}

.typing {
  border-right: 3px solid white;
  white-space: nowrap;
  overflow: hidden;
  width: 0;
  animation: typing 3s steps(30, end) forwards, blink 0.8s infinite;
}

@keyframes typing {
  from { width: 0 }
  to { width: 100% }
}

@keyframes blink {
  50% { border-color: transparent }
}
</style>
</head>

<body>
<div id="particles-box">
  <div id="particles-js"></div>

  <div class="overlay-text typing">
    ⚡ Predicting Words Before You Do
  </div>

  <div class="sub-text">
    Smart AI Autocomplete • Fast • Accurate • Futuristic
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 80},
    "color": {"value": ["#00ffff", "#ff00ff", "#ffffff"]},
    "shape": {"type": "circle"},
    "opacity": {"value": 0.6},
    "size": {"value": 4},
    "line_linked": {
      "enable": true,
      "distance": 120,
      "color": "#ffffff",
      "opacity": 0.3,
      "width": 1
    },
    "move": {"enable": true, "speed": 2}
  }
});
</script>
</body>
</html>
"""

components.html(particles_js, height=400)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_all()

# -----------------------------
# FUNCTIONS
# -----------------------------
def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(seq, verbose=0)
    return tokenizer.index_word.get(np.argmax(pred), "")

def predict_top_words(text, n=3):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')
    pred = model.predict(seq, verbose=0)[0]
    top = pred.argsort()[-n:][::-1]
    return [tokenizer.index_word.get(i, "") for i in top]

def typing_effect(text):
    placeholder = st.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(f"### 🧠 {output}")
        time.sleep(0.03)

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Next Word Prediction AI")
st.write("✨ Futuristic AI autocomplete with neon interface")

input_text = st.text_input("✍️ Enter your sentence:")

col1, col2 = st.columns(2)

# Predict
with col1:
    if st.button("🔮 Predict"):
        if input_text.strip():
            with st.spinner("Thinking... 🤔"):
                time.sleep(1)
                word = predict_next_word(input_text)
                top_words = predict_top_words(input_text)

            st.success(f"👉 Next Word: **{word}**")
            st.info(f"🔥 Suggestions: {', '.join(top_words)}")
        else:
            st.warning("⚠️ Enter text")

# Generate
with col2:
    num_words = st.slider("⚙️ Words to generate", 1, 10, 3)

    if st.button("⚡ Generate"):
        if input_text.strip():
            text = input_text
            with st.spinner("Generating... 🚀"):
                time.sleep(1)
                for _ in range(num_words):
                    text += " " + predict_next_word(text)

            typing_effect(text)
        else:
            st.warning("⚠️ Enter text")

# Footer
st.markdown("---")
st.markdown("🚀 Built by Satyam | LSTM + Streamlit + Premium UI")