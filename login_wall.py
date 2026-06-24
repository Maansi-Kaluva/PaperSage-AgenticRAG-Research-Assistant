import streamlit as st

def show_login():
    st.markdown("""
    <style>
    .login-wrapper {
        display: flex;
        justify-content: center;
        margin-top: 8vh;
    }

    .login-box {
        max-width: 420px;
        width: 100%;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        background: #0d2020;
        border: 1px solid #1a3a3a;
        border-radius: 12px;
    }

    .login-logo {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }

    .login-title {
        font-size: 1.5rem;
        font-weight: 650;
        color: #e0f2f1;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .login-subtitle {
        font-size: 0.85rem;
        color: #4db6ac;
        margin-bottom: 2rem;
    }

    .stTextInput > div > div > input {
        background-color: #0a1515 !important;
        border: 1px solid #1a3a3a !important;
        border-radius: 8px !important;
        color: #b2dfdb !important;
        padding: 0.6rem 0.9rem !important;
        font-size: 0.95rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00897b !important;
        box-shadow: 0 0 0 2px rgba(0, 137, 123, 0.15) !important;
    }

    .stTextInput label {
        color: #4db6ac !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    .stButton > button {
        width: 100% !important;
        background-color: #00695c !important;
        color: #e0f2f1 !important;
        border: 1px solid #00897b !important;
        border-radius: 8px !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.95rem !important;
        font-weight: 550 !important;
        margin-top: 0.5rem !important;
        transition: background-color 0.15s ease !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button:hover {
        background-color: #00796b !important;
        border-color: #4db6ac !important;
    }

    .error-msg {
        background: rgba(185, 50, 50, 0.12);
        border: 1px solid rgba(185, 50, 50, 0.3);
        border-radius: 7px;
        color: #e07070;
        font-size: 0.85rem;
        padding: 0.6rem 0.9rem;
        margin-top: 0.75rem;
    }

    .footer-note {
        text-align: center;
        color: #1a3a3a;
        font-size: 0.75rem;
        margin-top: 2rem;
        letter-spacing: 0.02em;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-wrapper">
        <div class="login-box">
            <div class="login-logo">📚</div>
            <div class="login-title">PaperSage</div>
            <div class="login-subtitle">Research Paper Assistant — sign in to continue</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter access password",
                label_visibility="visible",
            )
            login_clicked = st.button("Sign in →")

            if login_clicked:
                correct = st.secrets.get("APP_PASSWORD", "demo123")
                if password == correct:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.markdown(
                        '<div class="error-msg">Incorrect password. Please try again.</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown(
        '<div class="footer-note">PaperSage · built with LangGraph + Qdrant</div>',
        unsafe_allow_html=True,
    )


def require_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        show_login()
        st.stop()