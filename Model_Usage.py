#Model_Usage
import spacy

# =========================
# CONFIG
# =========================
MODEL_PATH = r"./trained_ner_model"  # path to your saved NER model

# =========================
# Load the trained model
# =========================
try:
    nlp = spacy.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model at {MODEL_PATH}: {e}")
    exit(1)

# =========================
# Interactive user input
# =========================
print("NER Model Loaded! Enter a sentence to extract entities (type 'exit' to quit).")

while True:
    text = input("\nEnter text: ")
    if text.strip().lower() == "exit":
        break

    doc = nlp(text)
    if doc.ents:
        print("\nDetected Entities:")
        for ent in doc.ents:
            print(f" - {ent.text} [{ent.label_}]")
    else:
        print("No entities found.")
