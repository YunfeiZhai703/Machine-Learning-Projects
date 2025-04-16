import numpy as np

A = np.array([
    [1, 2, 3],
    [1, 4, 2],
    [2, 3, 1],
    [3, 1, 2],
])

K = 2  # Number of mixture components
sd = np.array([0, 1, 0])  # Example random assignments
W = 4  # Maximum word ID is 4 in this example

swk = np.zeros((W, K))
sk_docs = np.zeros((K, 1), dtype=int)

D = np.max(A[:, 0])  # Number of documents in A (D=3)

# Populate the count matrices by looping over documents
for d in range(D):
    training_documents = np.where(A[:, 0] == d+1)  # Get all occurrences of document d in the training data
    w = np.array(A[training_documents, 1])  # Unique words in document d
    c = np.array(A[training_documents, 2])  # Counts of words in document d
    k = sd[d]  # Document d is in mixture k
    swk[w-1, k] += c  # Number of times w is assigned to component k
    sk_docs[k] += 1
    print(w)
    print(c)
    print(swk)

print("swk (Word Counts Assigned to Each Mixture Component):")
print(swk)

print("\nsk_docs (Number of Documents Assigned to Each Mixture Component):")
print(sk_docs)

sk_words = np.sum(swk, axis=0)
print(sk_words)