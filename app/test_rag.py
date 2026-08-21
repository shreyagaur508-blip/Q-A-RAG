from rag import generate_answer


relevant_chunks = [
    {
        "source": "machine_learning.pdf",
        "text": (
            "Supervised learning uses labeled data "
            "to train a model. Examples include "
            "classification and regression."
        )
    },
    {
        "source": "machine_learning.pdf",
        "text": (
            "Unsupervised learning works with "
            "unlabeled data. Clustering is a "
            "common example."
        )
    }
]


question = "What is supervised learning?"


print("Question:")
print(question)

print("\nGenerating answer...\n")


answer = generate_answer(
    question,
    relevant_chunks
)


print("Answer:")
print(answer)