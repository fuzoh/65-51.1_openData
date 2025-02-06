import requests
import pdf2image
import pytesseract
import nltk
import hashlib

nltk.download("punkt")


def clean_text(text):
    """Remove section titles and figure descriptions from text"""
    clean = "\n".join(
        [
            row
            for row in text.split("\n")
            if (len(row.split(" "))) > 3
            and not (row.startswith("(a)"))
            and not row.startswith("Figure")
        ]
    )
    return clean


pdf = requests.get("https://arxiv.org/pdf/2110.03526.pdf")
doc = pdf2image.convert_from_bytes(pdf.content)  # Get the article text
article = []

for page_number, page_data in enumerate(doc):
    txt = pytesseract.image_to_string(page_data).encode(
        "utf-8"
    )  # Sixth page are only references
    if page_number < 6:
        article.append(txt.decode("utf-8"))

article_txt = " ".join(article)

text = article_txt.split("INTRODUCTION")[1]
ctext = clean_text(text)
sentences = nltk.tokenize.sent_tokenize(ctext)