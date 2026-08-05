import pandas as pd

data = {
    "Word": ["The", "cat", "runs"],
    "POS": ["DT", "NN", "VB"]
}

df = pd.DataFrame(data)

print(df)
print("POS Tagging Completed")