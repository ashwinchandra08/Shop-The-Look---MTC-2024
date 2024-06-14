import streamlit as st

uploaded_file = st.file_uploader("Upload your image...", type=["jpg","jpeg","png","pdf"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
