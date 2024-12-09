from collections import defaultdict
import math


def naive_bayes_train(docs, labels, alpha=1):
    """
    Train a Naive Bayes classifier with Laplacian smoothing.

    Parameters:
        docs (list of lists): List of documents, each represented as a list of words.
        labels (list): List of labels corresponding to each document.
        alpha (float): Smoothing parameter (default is 1).

    Returns:
        tuple: Vocabulary, class probabilities, and conditional probabilities.
    """
    # Create vocabulary
    vocab = set(word for doc in docs for word in doc)
    vocab_size = len(vocab)

    # Count occurrences
    class_counts = defaultdict(int)
    word_counts = defaultdict(lambda: defaultdict(int))

    for doc, label in zip(docs, labels):
        class_counts[label] += 1
        for word in doc:
            word_counts[label][word] += 1

    total_docs = len(docs)

    # Calculate class probabilities
    class_probs = {label: count / total_docs for label, count in class_counts.items()}

    # Calculate conditional probabilities with Laplacian smoothing
    cond_probs = defaultdict(dict)
    for label in class_counts:
        total_words = sum(word_counts[label].values())
        for word in vocab:
            cond_probs[label][word] = (word_counts[label][word] + alpha) / (
                total_words + alpha * vocab_size
            )

    return vocab, class_probs, cond_probs


def naive_bayes_predict(doc, vocab, class_probs, cond_probs):
    """
    Predict the class of a document using a trained Naive Bayes classifier.

    Parameters:
        doc (list): List of words in the document.
        vocab (set): Vocabulary of the training data.
        class_probs (dict): Prior probabilities of each class.
        cond_probs (dict of dicts): Conditional probabilities of words given a class.

    Returns:
        str: Predicted class label.
    """
    max_prob = float("-inf")
    best_class = None

    for label, class_prob in class_probs.items():
        # Start with log of the class probability
        log_prob = math.log(class_prob)
        for word in doc:
            if word in vocab:  # Use only words in the vocabulary
                # Handle unseen words
                log_prob += math.log(cond_probs[label].get(word, 1e-6))
        if log_prob > max_prob:
            max_prob = log_prob
            best_class = label

    return best_class


# Example Data
documents = [
    ["buy", "cheap", "meds"],
    ["limited", "offer", "buy"],
    ["cheap", "viagra", "now"],
    ["hello", "how", "are", "you"],
    ["meeting", "today", "agenda"],
    ["let", "us", "schedule", "meeting"],
]
labels = ["Spam", "Spam", "spam", "Spam", "Not Spam", "Not Spam"]

# Train the model
vocab, class_probs, cond_probs = naive_bayes_train(documents, labels, alpha=1)


# Predict the class of a new document
def input_and_test(vocab, class_probs, cond_probs):
    """
    Takes user input for a new document, processes it, and predicts the class.
    """
    print("Enter a document as a space-separated string (e.g., 'cheap offer now'):")
    user_input = input().strip()
    new_doc = user_input.split()

    predicted_class = naive_bayes_predict(new_doc, vocab, class_probs, cond_probs)
    print(f"\nPredicted Class: {predicted_class}")


# Call the input function to test
input_and_test(vocab, class_probs, cond_probs)
