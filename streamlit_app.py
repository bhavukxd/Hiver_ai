import streamlit as st
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from generator import generate_reply

st.set_page_config(page_title="Email Reply Assistant", page_icon="")

st.title("Email Reply Assistant")

email = st.text_area("Customer Email", height=120, placeholder="Paste a customer email here...")

if st.button("Generate Reply", use_container_width=True):
    if not email.strip():
        st.error("Please enter a customer email.")
    else:
        with st.spinner("Generating reply..."):
            reply = generate_reply(email)
            st.markdown("**Suggested Reply**")
            st.write(reply)