# app.py
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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .urdu-text {
        font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.3rem;
        direction: rtl;
        text-align: right;
        line-height: 1.6;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #f0f8ff;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTextInput input {
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load the chatbot model with caching to avoid reloading on every interaction."""
    try:
        # Check if model files exist
        if not os.path.exists("best_transformer_bleu.pt"):
            st.error("❌ Model file 'best_transformer_bleu.pt' not found. Please ensure it's in the same directory.")
            return None
        if not os.path.exists("vocab.txt"):
            st.error("❌ Vocabulary file 'vocab.txt' not found. Please ensure it's in the same directory.")
            return None
            
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
        <h4>🔄 کیسے استعمال کریں:</h4>
        <p>• نیچے دیے گئے باکس میں اپنا پیغام لکھیں</p>
        <p>• 'بھیجیں' بٹن پر کلک کریں یا Enter دبائیں</p>
        <p>• بوٹ کا جواب اوپر نظر آئے گا</p>
        <br>
        <h4>🔄 How to use:</h4>
        <p>• Type your message in the box below</p>
        <p>• Click 'Send' or press Enter</p>
        <p>• Bot's response will appear above</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📝 مثالیں - Examples")
        
        # Initialize chatbot in session state
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
        
        # Example buttons
        examples = [
            "آپ کا نام کیا ہے؟",
            "کیا حال ہے؟", 
            "اسلام علیکم",
            "آپ کیسے ہیں؟",
            "شکریہ",
            "خدا حافظ"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                # Process example message
                st.session_state.messages.append({"role": "user", "content": example})
                if st.session_state.chatbot:
                    with st.spinner("🤖 بوٹ سوچ رہا ہے..."):
                        try:
                            bot_response = st.session_state.chatbot.generate_response(example)
                            st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        except Exception as e:
                            error_msg = f"معذرت، جواب دینے میں خرابی ہوئی۔ Sorry, error generating response: {str(e)}"
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "ماڈل دستیاب نہیں ہے۔ Model not available."
                    })
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <h4>📊 ماڈل کی معلومات</h4>
        <p><strong>🧠 ٹرانسفارمر ماڈل</strong></p>
        <p>• 2 انکوڈر/ڈیکوڈر پرت</p>
        <p>• 2 ہیڈز اٹینشن</p>
        <p>• 256 ایمبیڈنگ ڈائمینشن</p>
        <p>• 512 فیڈ فارورڈ ڈائمینشن</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button in sidebar
        if st.session_state.messages and len(st.session_state.messages) > 1:
            if st.button("🗑️ چیٹ صاف کریں - Clear Chat", use_container_width=True):
                st.session_state.messages = [
                    {
                        "role": "assistant", 
                        "content": "السلام علیکم! میں اردو چیٹ بوٹ ہوں۔ آپ کیسے مدد کر سکتی ہوں؟"
                    }
                ]
                st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
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
    
    # Chat input form
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "اپنا پیغام یہاں لکھیں - Type your message here:",
                key="user_input",
                placeholder="مثال: آپ کا نام کیا ہے؟",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "📤 بھیجیں", 
                use_container_width=True
            )
    
    # Handle form submission
    if submit_button and user_input.strip():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        
        # Generate bot response
        if st.session_state.chatbot:
            with st.spinner("🤖 بوٹ سوچ رہا ہے..."):
                try:
                    bot_response = st.session_state.chatbot.generate_response(user_input.strip())
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                except Exception as e:
                    error_msg = f"معذرت، جواب دینے میں خرابی ہوئی۔ Sorry, error generating response: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            error_msg = "ماڈل دستیاب نہیں ہے۔ Model not available."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
    
    # Display model status
    if 'chatbot' in st.session_state and st.session_state.chatbot:
        st.sidebar.success("✅ ماڈل لوڈ ہو گیا - Model loaded successfully")
    else:
        st.sidebar.error("❌ ماڈل لوڈ نہیں ہو سکا - Model failed to load")
        
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🤖 اردو ٹرانسفارمر چیٹ بوٹ - Urdu Transformer Chatbot<br>"
        "Built with Streamlit & PyTorch"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
