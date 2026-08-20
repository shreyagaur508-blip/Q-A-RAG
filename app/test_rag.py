from rag import ask_question


question = "What is supervised learning?"


answer, sources = ask_question(question)


print("\nQUESTION:")
print(question)


print("\nANSWER:")
print(answer)


print("\nSOURCES:")
for i, source in enumerate(sources):
    print("\n" + "=" * 60)
    print("SOURCE", i + 1)
    print("=" * 60)
    print(source["text"])
    print("\nDistance:", source["distance"])