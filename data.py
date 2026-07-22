import os

os.makedirs('data1', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Sample training sentences
train_data = [
    (1, "The", "DT"), (2, "cat", "NN"), (3, "sat", "VBD"), (4, "on", "IN"), (5, "the", "DT"), (6, "mat", "NN"), (7, ".", "."),
    (1, "A", "DT"), (2, "dog", "NN"), (3, "barked", "VBD"), (4, "at", "IN"), (5, "the", "DT"), (6, "cat", "NN"), (7, ".", "."),
    (1, "The", "DT"), (2, "dog", "NN"), (3, "ran", "VBD"), (4, "fast", "RB"), (5, ".", ".")
]
# Sample dev sentences
dev_data = [
    (1, "The", "DT"), (2, "cat", "NN"), (3, "ran", "VBD"), (4, "on", "the", "DT"), (5, "mat", "NN"), (6, ".", "."),
    (1, "A", "DT"), (2, "dog", "NN"), (3, "sat", "VBD"), (4, ".", ".")
]
# Write train file
with open('data1/train', 'w') as f:
    for row in train_data:
        f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
# Write dev file
with open('data1/dev', 'w') as f:
    for row in dev_data:
        f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
print("Sample datasets created successfully in 'data1/' folder!")