
import streamlit as st
import torch
import logging
from model import TransformerModel, load_vocab, encode_text, pad_seq, ids_to_sentence, generate_response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Urdu Chatbot", page_icon="🤖", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vocabulary and model
@st.cache_resource
def load_model_and_vocab():
    try:
        logger.info("Loading vocabulary...")
        stoi, itos = load_vocab("vocab.txt")
        logger.info(f"Vocabulary loaded: {len(itos)} tokens")

        logger.info("Initializing model...")
        model = TransformerModel(
            vocab_size=len(itos),
            d_model=256,
            enc_layers=2,
            dec_layers=2,
            n_heads=2,
            d_ff=512,
            dropout=0.1
        )
        logger.info("Loading model weights...")
        checkpoint = torch.load("best_transformer_bleu.pt", map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model loaded successfully")
        return model, stoi, itos
    except Exception as e:
        logger.error(f"Error loading model or vocabulary: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        raise e

try:
    model, stoi, itos = load_model_and_vocab()
except Exception as e:
    st.stop()

# Title and description
st.title("🤖 Urdu Chatbot")
st.markdown("Type in Urdu to chat with the bot. It reconstructs sentences from corrupted inputs. Type 'exit', 'quit', or 'bye' to clear the chat.")

# Chat interface
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("👤 You:", placeholder="Type your message in Urdu...")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        if user_input.lower() in ['exit', 'quit', 'bye']:
            st.session_state.messages = []
            st.success("Chat cleared! Start a new conversation.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            try:
                # Generate response
                response = generate_response(user_input, model, stoi, itos)
                # Add bot response to chat history
                st.session_state.messages.append({"role": "bot", "content": response})
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error(f"Error generating response: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**👤 You**: {message['content']}")
    else:
        st.markdown(f"**🤖 Bot**: {message['content']}")

# Instructions
st.markdown("---")
st.markdown(f"**Instructions**: Enter Urdu text. The bot will respond with a reconstructed sentence. Type 'exit' to clear the chat. Deployed on {st.session_state.get('deploy_date', 'October 24, 2025, 02:47 PM PKT')}.")
```

