"""Testmodul zur Überprüfung der KI-gestützten Vorverarbeitung von Interview-Transkripten."""

import json
from typing import List
from preprocessor import Preprocessor, Interview

preprocessor : Preprocessor = Preprocessor()

# read the questions_file from data/questions.json and make a List[str]
with open("data/questions.json", "r", encoding="utf-8") as json_file:
    questions : List[str] = json.load(json_file)

with open("data/sample_transcript.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

interview : Interview = preprocessor.ai_preprocess(
    text=transcript,
    augmented_questions=questions,
    use_multithreading=True,
    threads=2
)

print(len(interview.snippets))
print(interview.speakers)
print(interview.questions)
