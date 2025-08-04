import spacy
from spacy.training import Example, offsets_to_biluo_tags
from spacy.util import minibatch, compounding
import json
import random
import os
from pathlib import Path
from datetime import datetime


def has_overlap(entities):
    """Check if any entity spans overlap."""
    sorted_ents = sorted(entities, key=lambda x: x[0])
    for i in range(1, len(sorted_ents)):
        if sorted_ents[i][0] < sorted_ents[i - 1][1]:
            return True
    return False


def is_entity_alignment_valid(text, entities):
    """Check if entities align with token boundaries."""
    try:
        dummy_nlp = spacy.blank("en")
        doc = dummy_nlp.make_doc(text)
        tags = offsets_to_biluo_tags(doc, entities)
        return '-' not in tags
    except Exception as e:
        print(f"Alignment issue: {e}")
        return False


def load_training_data(json_path):
    with open(json_path, "r", encoding="utf8") as f:
        raw_data = json.load(f)

    training_data = []
    for i, item in enumerate(raw_data):
        if isinstance(item, dict) and "text" in item and "entities" in item:
            text = item["text"]
            entities = item["entities"]
        elif isinstance(item, list) and len(item) == 2:
            text, ann = item
            entities = ann.get("entities", [])
        else:
            print(f" Invalid format at index {i}. Skipping.")
            continue

        if not text.strip() or not entities:
            continue
        if has_overlap(entities):
            print(f"Overlapping entities at index {i}. Skipping.")
            continue
        if not is_entity_alignment_valid(text, entities):
            print(f" Misaligned entities at index {i}. Skipping.")
            continue

        training_data.append((text, {"entities": entities}))

    return training_data


def evaluate_model(nlp, train_data):
    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annotations))
    return nlp.evaluate(examples)


def train_custom_ner(json_file="labeled_resume_data.json", output_dir="custom_ner_model", iterations=30):
    train_data = load_training_data(json_file)
    if not train_data:
        print(" No valid training data found.")
        return

    print(f"Loaded {len(train_data)} valid training examples.")

    # Initialize blank English model
    nlp = spacy.blank("en")

    # Create or get NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add all labels
    labels = set()
    for _, ann in train_data:
        for start, end, label in ann["entities"]:
            ner.add_label(label)
            labels.add(label)

    print(f" Labels: {sorted(labels)}")

    # Training
    with nlp.disable_pipes(*[p for p in nlp.pipe_names if p != "ner"]):
        optimizer = nlp.begin_training()
        for i in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.5))
            for batch in batches:
                examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
                nlp.update(examples, sgd=optimizer, losses=losses)
            print(f"📘 Iteration {i + 1}/{iterations} - Loss: {losses['ner']:.4f}")

    # Save model with versioned folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(output_dir) / f"model_{timestamp}"
    model_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(str(model_path))
    print(f" Model saved to: {model_path}")

    # Optional: Evaluate on training data
    metrics = evaluate_model(nlp, train_data)
    print("\n Evaluation on training set:")
    print(f"Precision: {metrics['ents_p']:.3f}")
    print(f"Recall:    {metrics['ents_r']:.3f}")
    print(f"F-score:   {metrics['ents_f']:.3f}")


# 🔽 Run this
if __name__ == "__main__":
    train_custom_ner("cleaned_annotated_resumes.json", iterations=50)
