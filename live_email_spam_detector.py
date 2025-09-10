import imaplib
import email
import time
import joblib
from spam_detector import clean_text  # your clean_text function

# -----------------------------
# CONFIG
# -----------------------------
IMAP_HOST = 'imap.gmail.com'
EMAIL_USER = 'your_email@gmail.com'
EMAIL_PASS = 'your_app_password'  # Use App Password
CHECK_INTERVAL = 300  # seconds
SPAM_THRESHOLD = 0.8

# Load trained model
pipeline = joblib.load("models/spam_detector_pipeline.joblib")

# -----------------------------
# CONNECT TO MAILBOX
# -----------------------------
mail = imaplib.IMAP4_SSL(IMAP_HOST)
mail.login(EMAIL_USER, EMAIL_PASS)
mail.select('inbox')

print("Connected to mailbox. Checking emails every 5 minutes...\n")

# -----------------------------
# FUNCTION TO PROCESS EMAILS
# -----------------------------
def process_email(email_id):
    status, msg_data = mail.fetch(email_id, '(RFC822)')
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    # Extract plain text
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body += part.get_payload(decode=True).decode(errors='ignore')
    else:
        body = msg.get_payload(decode=True).decode(errors='ignore')

    # Clean text and predict
    clean_body = clean_text(body)
    prob = pipeline.predict_proba([clean_body])[0]
    label = 'spam' if prob[1] >= SPAM_THRESHOLD else 'ham'

    print(f"From: {msg['From']}")
    print(f"Subject: {msg['Subject']}")
    print(f"Predicted: {label} (spam prob: {prob[1]:.2f})")
    print("-" * 50)

    # Optional: Move spam emails to Gmail Spam folder
    if label == 'spam':
        mail.store(email_id, '+X-GM-LABELS', 'Spam')  # Gmail specific

# -----------------------------
# LOOP TO CHECK EMAILS
# -----------------------------
while True:
    status, messages = mail.search(None, 'UNSEEN')  # only unread emails
    email_ids = messages[0].split()

    if email_ids:
        print(f"{len(email_ids)} new email(s) found.\n")
        for eid in email_ids:
            process_email(eid)
    else:
        print("No new emails.")

    time.sleep(CHECK_INTERVAL)
