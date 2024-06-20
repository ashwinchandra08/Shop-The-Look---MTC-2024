import streamlit as st
import copy
import base64
import requests
import json

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
    current_image = copy.deepcopy(encoded_image)
    current_image = base64.b64decode(current_image)

    if st.button('Analyze Image'):
        try:
            # Send POST request to backend service
            response = requests.post("http://localhost:5000/analyze-image", json={"image": encoded_image})

            if response.status_code == 200:
                result = response.json()
                st.success("Image analyzed successfully!")
                
                # Process and display chat message content
                choices = result.get('response', {}).get('choices', [])
                if choices:
                    for choice in choices:
                        with c.chat_message(choice["message"]["role"]):
                            st.markdown(choice["message"]["content"].strip())
                else:
                    st.error("No items found in the analysis response.")
            else:
                error_response = response.json()
                st.error(f"Error: {error_response.get('error', 'Unknown error')}")
                st.write(error_response)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            print(error_msg)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with c.chat_message(message["role"]):
        st.markdown(message["content"]["text"])  # Print only the text content of the message
        if "image" in message:
            st.image(message["image"], width=100)
    
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with c.chat_message("user"):
        st.write(prompt)
        if uploaded_file is not None:
            st.image(current_image, width=100)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": {"text": prompt}, "image": current_image})
