import imaplib
import email
import time
import joblib
import re, string

# -----------------------------
# CONFIG
# -----------------------------
IMAP_HOST = 'imap.gmail.com'
EMAIL_USER = 'logobuddy98@gmail.com'
EMAIL_PASS = 'knvo jyki plow sekj'  # Use App Password
CHECK_INTERVAL = 300  # seconds
SPAM_THRESHOLD = 0.4

# Load trained model
pipeline = joblib.load("models/spam_detector_pipeline.joblib")

# -----------------------------
# CONNECT TO MAILBOX
# -----------------------------
mail = imaplib.IMAP4_SSL(IMAP_HOST)
mail.login(EMAIL_USER, EMAIL_PASS)
mail.select('inbox')

print("Connected to mailbox. Checking emails every 5 minutes...\n")

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

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
    spam_prob = pipeline.predict_proba([clean_body])[0][1]  # probability of spam
    label = 'spam' if spam_prob >= SPAM_THRESHOLD else 'ham'

    print(f"From: {msg['From']}")
    print(f"Subject: {msg['Subject']}")
    # print(f"Predicted: {label} (spam prob: {prob[1]:.2f})")
    print(f"Predicted: {label}")
    print("-" * 50)

    # Optional: Move spam emails to Gmail Spam folder
    if label == 'spam':
     mail.store(email_id, '+X-GM-LABELS', '\\Spam')  # Gmail system label


# -----------------------------
# FETCH UNREAD EMAILS ONCE
# -----------------------------
status, messages = mail.search(None, 'UNSEEN')
email_ids = messages[0].split()

# Take only the last 10 emails
email_ids = email_ids[-5:]

if email_ids:
    print(f"{len(email_ids)} new email(s) found (max 5).")
    for eid in email_ids:
        process_email(eid)
else:
    print("No new emails.")

# Close connection after processing
mail.logout()
print("\nDone reading emails.")
