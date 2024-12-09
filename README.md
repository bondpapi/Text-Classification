# Text-Classification
This project implements a simple Naive Bayes Classifier for text classification using Laplacian Smoothing. The classifier predicts whether a given text belongs to the Spam or Not Spam class based on training data.

## Features

- Training: Processes labeled text data to compute class probabilities and conditional probabilities using Laplacian Smoothing.

- Prediction: Predicts the class of new text inputs based on the trained model.

- Interactive Input: Allows users to test the model with their own text inputs.

- Simple Implementation: Designed to be lightweight and easy to understand.

## How It Works

1. **Training Phase:** 

    The model calculates:

    - *Prior probabilities:* The likelihood of each class based on the training data.

    - *Conditional probabilities:* The likelihood of each word appearing in a given class, smoothed using Laplacian Smoothing.
 
2. **Prediction Phase:**

    - For a new text input, the model calculates the posterior probabilities for each class and predicts the class with the highest probability.

## Laplacian Smoothing

Laplacian Smoothing adds a small constant
α (default is 1) to the word counts during training. This prevents zero probabilities for unseen words, ensuring the model remains robust to sparse data.

## Installation

1. Clone the repository or copy the script to your local machine.

2. Ensure you have Python 3.x installed.

## Usage

**Run the Script:**
```python 
spam.py
```

**Interactive Input:**

 - Enter a space-separated string of words when prompted.
 - Example input: ```cheap offer now.```

**Output:**

- The script predicts whether the input text belongs to the "Spam" or "Not Spam" class.

## Example

### Training Data:

The training dataset includes the following documents:


| Text                          | Classification  |
|:-----------------------------:|:---------------:|
| ```buy cheap meds```          | Spam            |
| ```limited offer buy```       | Spam            |
| ```cheap viagra now```        | Spam            |
| ```hello how are you```       | Spam            |
|```Today's Meeting Agenda```   | Not Spam        |
|```Let us schedule a meeting```| Not Spam        |

## Testing:

Example interaction:

```plaintext
Enter a document as a space-separated string(e.g.'cheap offer now'): cheap offer now

Predicted Class: Spam
```    
## Code Overview

**Key Functions** 

1. ```naive_bayes_train (docs, labels, alpha=1):```

    Trains the Naive Bayes model using the provided documents and labels.

    Computes prior probabilities and conditional probabilities with Laplacian smoothing.

2. ```naive_bayes_predict(doc, vocab, class_probs, cond_probs):```

    Predicts the class of a new document based on the trained model.

3. ```input_and_test(vocab, class_probs, cond_probs):```

    Allows the user to input text interactively for testing.

## Customization

Modify the ```documents``` and ```labels``` variables to use your own dataset.

Adjust the smoothing parameter α in the ```naive_bayes_train``` function to experiment with different levels of smoothing.

## Requirements

Python 3.x

No additional libraries required.
