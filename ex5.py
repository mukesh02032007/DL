import re
import pandas as pd
import numpy as np
import math
import nltk

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Load dataset
data = pd.read_csv("IMDBDataset.csv")


# Function to clean HTML tags, URLs, and special characters
def remove_tags(string):
    removelist = ""
    result = re.sub(r"<.*?>", "", string)          # Remove HTML tags
    result = re.sub(r"https://.*", "", result)     # Remove URLs
    result = re.sub(r"[^a-zA-Z0-9" + removelist + r"]", " ", result)
    result = result.lower()
    return result


# Apply preprocessing
data["review"] = data["review"].apply(remove_tags)

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Remove stopwords
stop_words = set(stopwords.words("english"))

data["review"] = data["review"].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

# Lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    return " ".join(
        [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    )


data["review"] = data["review"].apply(lemmatize_text)

# Prepare data
reviews = data["review"].values
labels = data["sentiment"].values

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Train/Test split
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    reviews,
    encoded_labels,
    stratify=encoded_labels
)

# Tokenization and Padding
vocab_size = 3000
embedding_dim = 100
max_length = 200
oov_tok = "<OOV>"
padding_type = "post"
trunc_type = "post"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(
    train_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(
    test_sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(24, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# Train model
num_epochs = 5

history = model.fit(
    train_padded,
    train_labels,
    epochs=num_epochs,
    verbose=1,
    validation_split=0.1
)

# Evaluate on test set
prediction = model.predict(test_padded)
pred_labels = [1 if i >= 0.5 else 0 for i in prediction]

print("Accuracy on test set:", accuracy_score(test_labels, pred_labels))

print(
    classification_report(
        test_labels,
        pred_labels,
        target_names=encoder.classes_
    )
)

# Predict sentiment on new reviews
sentences = [
    "The movie was very touching and heartwarming",
    "I have never seen a terrible movie like this",
    "The movie plot is terrible but it had good acting"
]

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(
    sequences,
    maxlen=max_length,
    padding=padding_type,
    truncating=trunc_type
)

prediction = model.predict(padded)
pred_labels = [1 if i >= 0.5 else 0 for i in prediction]

# Output predictions
for i in range(len(sentences)):
    print(f"\nReview: {sentences[i]}")
    print(
        "Predicted Sentiment:",
        "Positive" if pred_labels[i] == 1 else "Negative"
    )
