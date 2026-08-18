from pdf_loader import extract_text_from_pdf
from chunker import create_chunks


pdf_path = "data/documents/machine_learning.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = create_chunks(text)


print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n" + "=" * 50)
    print("CHUNK", i + 1)
    print("=" * 50)
    print(chunk)