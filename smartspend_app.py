import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import joblib
from PIL import Image
import numpy as np
import easyocr
import re
import pytesseract
import cv2

# ------------------ DATABASE SETUP ------------------
conn = sqlite3.connect('expenses.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    category TEXT,
    description TEXT,
    amount REAL
)
""")
conn.commit()

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="SmartSpend", layout="centered")
st.title("💸 SmartSpend - AI Expense Tracker")

# ------------------ LOAD ML MODEL ------------------
@st.cache_resource
def load_model():
    return joblib.load("category_model.pkl")

model = load_model()

# ------------------ SESSION STATE FOR OCR PREFILL ------------------
if 'ocr_description' not in st.session_state:
    st.session_state['ocr_description'] = ''
if 'ocr_amount' not in st.session_state:
    st.session_state['ocr_amount'] = 0.0

# ------------------ EXPENSE FORM ------------------
st.header("➕ Add New Expense")

with st.form("expense_form"):
    date = st.date_input("Date", datetime.today())
    description = st.text_input("Description", value=st.session_state['ocr_description'])
    amount = st.number_input("Amount (₹)", min_value=0.0, step=0.5, value=st.session_state['ocr_amount'])

    # Predict category
    predicted_category = model.predict([description])[0] if description else "Others"
    st.markdown(f"**Predicted Category:** `{predicted_category}`")

    submitted = st.form_submit_button("Add Expense")
    if submitted:
        cursor.execute("INSERT INTO expenses (date, category, description, amount) VALUES (?, ?, ?, ?)",
                       (str(date), predicted_category, description, amount))
        conn.commit()
        st.success("✅ Expense added successfully!")
        st.session_state['ocr_description'] = ''
        st.session_state['ocr_amount'] = 0.0

# ------------------ OCR RECEIPT SCANNER ------------------
st.header("📷 Scan a Receipt")

uploaded_file = st.file_uploader("Upload a receipt image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Resize large image
    MAX_WIDTH = 800
    if image.width > MAX_WIDTH:
        scale = MAX_WIDTH / image.width
        new_size = (MAX_WIDTH, int(image.height * scale))
        image = image.resize(new_size)

    st.image(image, caption="Uploaded Receipt", use_container_width=True)

    with st.spinner("🔍 Scanning for text (OCR)..."):
        try:
            import pytesseract
            import cv2

            # Convert to OpenCV format and preprocess
            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # OCR using pytesseract
            extracted_text = pytesseract.image_to_string(thresh)
            st.markdown(f"📝 **Extracted Text:** `{extracted_text}`")

            # Try to extract amount
            import re
            amounts = re.findall(r'₹\s?\d+\.?\d*', extracted_text) or re.findall(r'\d+\.\d{2}', extracted_text)
            detected_amount = amounts[0] if amounts else "Not found"
            st.markdown(f"💰 **Detected Amount:** `{detected_amount}`")

            if st.button("📥 Use This in Form"):
                st.session_state['ocr_description'] = extracted_text
                try:
                    st.session_state['ocr_amount'] = float(re.sub("[^0-9.]", "", detected_amount))
                except:
                    st.session_state['ocr_amount'] = 0.0
                st.experimental_rerun()

        except Exception as e:
            st.error(f"❌ OCR failed: {str(e)}")

# ------------------ EXPENSE HISTORY ------------------
st.header("📒 Expense History")

df = pd.read_sql_query("SELECT * FROM expenses ORDER BY date DESC", conn)
st.dataframe(df, use_container_width=True)

# ------------------ SUMMARY ------------------
st.header("📊 Summary")

if not df.empty:
    total = df['amount'].sum()
    st.subheader(f"💰 Total Spent: ₹{total:.2f}")

    category_summary = df.groupby("category")['amount'].sum().reset_index()
    st.bar_chart(category_summary.set_index("category"))
else:
    st.info("No expenses yet. Add some above!")
