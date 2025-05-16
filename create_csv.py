import pandas as pd

# Read the original CSVs
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Add labels
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Combine and keep only useful columns
df = pd.concat([fake, true], ignore_index=True)
df = df[['title', 'text', 'label']]

# Save combined file
df.to_csv('fake_or_real_news.csv', index=False)
print("✅ fake_or_real_news.csv created successfully.")
