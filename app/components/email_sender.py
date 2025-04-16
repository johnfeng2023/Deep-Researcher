import streamlit as st
from typing import Optional, Dict, Any

from app.utils.email_utils import send_research_email
from app.utils.config import config

def render_email_form(research_query: str, research_result: str) -> None:
    """
    Render a form to send research results via email.
    
    Args:
        research_query: The original research query
        research_result: The research result to send
    """
    st.subheader("📧 Send Results via Email")
    
    if not config.is_api_configured("email"):
        st.warning("Email configuration is not set up. Please add your email credentials to the .env file.")
        
        st.markdown("""
        ### How to Configure Email
        
        To enable email functionality, add the following to your `.env` file:
        
        ```
        EMAIL_USERNAME=your_email@gmail.com
        EMAIL_PASSWORD=your_app_password
        EMAIL_SMTP_SERVER=smtp.gmail.com
        EMAIL_SMTP_PORT=587
        ```
        
        **Note for Gmail users:** You'll need to use an App Password instead of your regular password. 
        [Learn more about App Passwords](https://support.google.com/accounts/answer/185833)
        """)
        
        return
    
    with st.form("email_form"):
        st.subheader("Send Research Results")
        
        recipient_email = st.text_input(
            "Recipient Email",
            help="Email address to send the research results to"
        )
        
        subject = st.text_input(
            "Subject",
            value=f"Research Results: {research_query[:50]}{'...' if len(research_query) > 50 else ''}",
            help="Subject line for the email"
        )
        
        send_button = st.form_submit_button("Send Email")
        
        if send_button:
            if not recipient_email:
                st.error("Please enter a recipient email address.")
                return
            
            with st.spinner("Sending email..."):
                result = send_research_email(
                    recipient_email=recipient_email,
                    subject=subject,
                    research_query=research_query,
                    research_result=research_result
                )
            
            if result.get("status") == "success":
                st.success(result.get("message", "Email sent successfully!"))
            else:
                st.error(result.get("message", "Failed to send email. Please check the configuration.")) 