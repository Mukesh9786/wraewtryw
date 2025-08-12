import streamlit as st
from dotenv import load_dotenv
from rag_chain import load_json_as_docs, create_vectorstore, build_custom_chain

load_dotenv()

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")

@st.cache_resource
def setup_chain():
    docs = load_json_as_docs("data.json")
    vectorstore = create_vectorstore(docs)
    return build_custom_chain(vectorstore)

qa_chain = setup_chain()

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

if "pending_intent" not in st.session_state:
    st.session_state.pending_intent = None

if "edit_index" not in st.session_state:
    st.session_state.edit_index = -1

# --- Custom CSS ---
st.markdown("""
    <style>
    .edit-button-container {
        display: flex;
        justify-content: flex-end;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .edit-button {
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #ccc;
        background-color: white;
        cursor: pointer;
    }
    .edit-button:hover {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Find the index of the last user message ---
last_user_message_index = -1
for i in range(len(st.session_state.chat_history) - 1, -1, -1):
    if st.session_state.chat_history[i]["role"] == "user":
        last_user_message_index = i
        break

# --- Display chat history ---
for idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    # Show edit button under the last user message
    if idx == last_user_message_index and not st.session_state.edit_mode:
        col1, col2 = st.columns([0.8, 0.2])
        with col2:
            if st.button("✏️ Edit", key="edit_btn"):
                st.session_state.edit_mode = True
                st.session_state.edit_index = idx
                st.rerun()

# --- Edit Mode ---
if st.session_state.edit_mode and st.session_state.edit_index != -1:
    original_prompt = st.session_state.chat_history[st.session_state.edit_index]["content"]
    edited_prompt = st.text_area("Edit your last question:", original_prompt)

    if st.button("Save & Regenerate"):
        # Remove the last user + assistant messages
        del st.session_state.chat_history[st.session_state.edit_index:]

        # Append edited user prompt
        st.chat_message("user").markdown(edited_prompt)
        st.session_state.chat_history.append({"role": "user", "content": edited_prompt})

        # Get new assistant response
        with st.chat_message("assistant"):
            with st.spinner("Regenerating..."):
                response = qa_chain(edited_prompt)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Reset edit state
        st.session_state.edit_mode = False
        st.session_state.edit_index = -1
        st.rerun()

# --- New Input Prompt ---
elif prompt := st.chat_input("Ask anything about your policy or general question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if prompt.strip().lower() == "yes" and st.session_state.pending_intent:
                response = qa_chain(prompt, st.session_state.pending_intent)
                st.session_state.pending_intent = None
            else:
                response = qa_chain(prompt)
                if "Are you looking for" in response and "Please reply 'Yes'" in response:
                    st.session_state.pending_intent = "WRStatus_NOT_COMPLETED"
                else:
                    st.session_state.pending_intent = None

            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.rerun()

