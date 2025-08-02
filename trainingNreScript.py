import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random

def fix_spans(text, entities, nlp):
    """Adjust entity spans to token boundaries to avoid misalignment."""
    doc = nlp.make_doc(text)
    fixed_entities = []

    for start, end, label in entities:
        token_start = None
        token_end = None

        # Find token start boundary (the token containing start char)
        for token in doc:
            if token.idx <= start < token.idx + len(token):
                token_start = token.idx
                break

        # Find token end boundary (the token containing end char)
        for token in reversed(doc):
            if token.idx < end <= token.idx + len(token):
                token_end = token.idx + len(token)
                break

        if token_start is None or token_end is None or token_start >= token_end:
            print(f"Warning: skipping entity '{text[start:end]}' due to bad alignment")
            continue

        fixed_entities.append((token_start, token_end, label))

    return fixed_entities

def train_ner_model():
    # Create blank English model
    nlp = spacy.blank("en")

    TRAIN_DATA = [
        (
            "John Smith graduated from MIT in 2020.",
            {"entities": [(0, 10, "PERSON"), (24, 27, "ORG"), (31, 35, "DATE")]}
        ),
        (
            "Jane Doe is a Python developer.",
            {"entities": [(0, 8, "PERSON"), (15, 21, "SKILL")]}
        ),
        # Add more examples as needed...
    ]

    # Add NER pipe if missing
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add entity labels to NER
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Fix spans and prepare training data
    fixed_train_data = []
    for text, annotations in TRAIN_DATA:
        ents = annotations.get("entities")
        fixed_ents = fix_spans(text, ents, nlp)
        if fixed_ents:
            fixed_train_data.append((text, {"entities": fixed_ents}))
        else:
            print(f"Skipping example due to no aligned entities: {text}")

    # Disable other pipes during training to speed it up
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(30):
            random.shuffle(fixed_train_data)
            losses = {}
            batches = minibatch(fixed_train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    examples.append(Example.from_dict(doc, annotations))
                nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")

    # Save the trained model to disk
    nlp.to_disk("custom_ner_model")
    print("Training complete! Model saved to ./custom_ner_model")

if __name__ == "__main__":
    train_ner_model()
