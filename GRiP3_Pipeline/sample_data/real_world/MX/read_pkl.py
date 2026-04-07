import pickle

# Replace 'your_file.pkl' with your actual file path
with open('meta_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
print(data)