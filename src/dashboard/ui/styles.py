import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
        /* FONDO GENERAL DE LA APP */
        .stApp {
            background-color: #0F172A;
        }
        
        /* Ajustes globales de scrollbar si se desean */
        ::-webkit-scrollbar {
            height: 8px;
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0F172A; 
        }
        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)