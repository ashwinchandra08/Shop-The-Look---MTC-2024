import streamlit as st
import copy
import base64

st.title("🛍️ Shop the Look")
st.caption("Upload an image and find similar items in our catalog")

c = st.container(height=300)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
uploaded_file = st.file_uploader("Upload your image...", type=["jpg","jpeg","png","pdf"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", width=100)
    
    encoded_image = base64.b64encode(uploaded_file.read()).decode("utf-8") 
    encoded_image_copy = copy.deepcopy(encoded_image)
    current_image = base64.b64decode(encoded_image_copy)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with c.chat_message(message["role"]):
        st.markdown(message["content"])
        if current_image is not None:
            st.image(message["image"], width=100)
    
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with c.chat_message("user"):
        st.write(prompt)
        if uploaded_file is not None:
            st.image(current_image, width=100)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "image": current_image})

