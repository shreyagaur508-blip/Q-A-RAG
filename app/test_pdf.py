from pdf_loader import extract_text_from_pdf


pdf_path = "data/documents/machine_learning.pdf"

text = extract_text_from_pdf(pdf_path)

print(text[:3000])