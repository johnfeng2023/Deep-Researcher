import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict

from app.utils.config import config

def send_research_email(
    recipient_email: str,
    subject: str,
    research_query: str,
    research_result: str,
    sender_email: Optional[str] = None,
) -> Dict[str, str]:
    """
    Send research results to the specified email address.
    
    Args:
        recipient_email: Email address to send the results to
        subject: Subject of the email
        research_query: The original research query
        research_result: The research results to send
        sender_email: Optional sender email (defaults to config email)
    
    Returns:
        Dictionary with status and message
    """
    if not config.is_api_configured("email"):
        return {
            "status": "error",
            "message": "Email is not configured. Please set up email credentials in your .env file."
        }
    
    # Set up email parameters
    sender = sender_email or config.email_config.username
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient_email
    
    # Create HTML content
    html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                .query {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
                .results {{ margin-top: 20px; }}
                .footer {{ margin-top: 40px; font-size: 0.9em; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Research Results</h1>
            
            <h2>Your Research Query:</h2>
            <div class="query">
                <p>{research_query}</p>
            </div>
            
            <h2>Research Findings:</h2>
            <div class="results">
                <p>{research_result.replace('\n', '<br/>')}</p>
            </div>
            
            <div class="footer">
                <p>This email was sent by Deep Researcher, an AI-powered research assistant.</p>
            </div>
        </body>
    </html>
    """
    
    # Attach HTML and plain text versions
    part1 = MIMEText(research_result, "plain")
    part2 = MIMEText(html, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # Send email
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(config.email_config.smtp_server, config.email_config.smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(config.email_config.username, config.email_config.password)
            server.send_message(msg)
        
        return {
            "status": "success",
            "message": f"Email sent successfully to {recipient_email}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to send email: {str(e)}"
        } 