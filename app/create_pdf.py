from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


pdf_path = "data/documents/machine_learning.pdf"

pdf = canvas.Canvas(pdf_path, pagesize=A4)

text = pdf.beginText(50, 800)
text.setFont("Helvetica", 12)

content = [
    "Machine Learning Basics",
    "",
    "Machine learning is a branch of artificial intelligence.",
    "It allows computers to learn patterns from data and make",
    "predictions or decisions without being explicitly programmed.",
    "",
    "There are three common types of machine learning:",
    "1. Supervised learning",
    "2. Unsupervised learning",
    "3. Reinforcement learning",
    "",
    "Supervised learning uses labeled data to train a model.",
    "Examples include classification and regression.",
    "",
    "Unsupervised learning works with unlabeled data.",
    "Clustering is a common example.",
    "",
    "Reinforcement learning allows an agent to learn by",
    "interacting with an environment and receiving rewards.",
    "",
    "Python is widely used for machine learning.",
]

for line in content:
    text.textLine(line)

pdf.drawText(text)
pdf.save()

print("PDF created successfully!")