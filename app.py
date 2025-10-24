import streamlit as st
import torch
from model import UrduChatbot
import os

# Page configuration
st.set_page_config(
    page_title="اردو چیٹ بوٹ - Urdu Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .urdu-text {
        font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.2rem;
        direction: rtl;
        text-align: right;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #1f77b4;
    }
    .bot-message {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff6b6b;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load the chatbot model with caching to avoid reloading on every interaction."""
    try:
        chatbot = UrduChatbot()
        return chatbot
    except Exception as e:
        st.error(f"ماڈل لوڈ کرنے میں خرابی: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 اردو چیٹ بوٹ - Urdu Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ رہنمائی - Instructions")
        st.markdown("""
        <div class="info-box">
        <h4>کیسے استعمال کریں:</h4>
        <p>1. نیچے دیے گئے باکس میں اپنا پیغام لکھیں</p>
        <p>2: 'بھیجیں' بٹن پر کلک کریں یا Enter دبائیں</p>
        <p>3: بوٹ کا جواب اوپر نظر آئے گا</p>
        <br>
        <h4>How to use:</h4>
        <p>1. Type your message in the box below</p>
        <p>2. Click 'Send' or press Enter</p>
        <p>3. Bot's response will appear above</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📝 مثالیں - Examples")
        examples = [
            "کیا حال ہے؟",
            "آپ کا نام کیا ہے؟",
            "میں ٹھیک ہوں",
            "اسلام علیکم",
            "آپ کیسے ہیں؟"
        ]
        for example in examples:
            if st.button(example, key=example):
                st.session_state.user_input = example
        
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <h4>📊 ماڈل کی معلومات</h4>
        <p><strong>ٹرانسفارمر ماڈل</strong></p>
        <p>• 2 انکوڈر/ڈیکوڈر پرت</p>
        <p>• 2 ہیڈز اٹینشن</p>
        <p>• 256 ایمبیڈنگ ڈائمینشن</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("🔄 ماڈل لوڈ ہو رہا ہے... Loading model..."):
            st.session_state.chatbot = load_chatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "السلام علیکم! میں اردو چیٹ بوٹ ہوں۔ آپ کیسے مدد کر سکتی ہوں؟"
        })
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>👤 آپ:</strong><br>
                    <div class="urdu-text">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 بوٹ:</strong><br>
                    <div class="urdu-text">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "اپنا پیغام یہاں لکھیں - Type your message here:",
            key="user_input",
            placeholder="اپنا پیغام یہاں لکھیں... Type your message here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 بھیجیں - Send", use_container_width=True)
    
    # Handle user input
    if (user_input and (send_button or st.session_state.get('user_input_trigger', False))) or (send_button and user_input):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate bot response
        if st.session_state.chatbot:
            with st.spinner("🤖 بوٹ سوچ رہا ہے... Bot is thinking..."):
                try:
                    bot_response = st.session_state.chatbot.generate_response(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                except Exception as e:
                    error_msg = f"معذرت، جواب دینے میں خرابی ہوئی۔ Sorry, error generating response: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            error_msg = "ماڈل دستیاب نہیں ہے۔ Model not available."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear input and trigger rerun
        st.session_state.user_input = ""
        st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ چیٹ صاف کریں - Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "السلام علیکم! میں اردو چیٹ بوٹ ہوں۔ آپ کیسے مدد کر سکتی ہوں؟"
                }
            ]
            st.rerun()

if __name__ == "__main__":
    main()
