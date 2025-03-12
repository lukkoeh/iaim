"""Module for preprocessing interview transcripts with AI support.
This module processes large transcripts into smaller parts and processes them with AI.
The results are converted into an Interview object.
The module was refactored with the help of Cursor to achieve a 10/10 pylint score.
Pre-Commit Hooks are recommended.
"""

import logging
import os
import re
import concurrent.futures
from typing import Optional, List
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
import tiktoken


class Snippet(BaseModel):
    """Represents a single conversation segment with speaker and text."""
    speaker: str
    text: str


class Interview(BaseModel):
    """Represents a complete interview with transcript, speakers, questions and snippets."""
    transcript: str
    speakers: List[str]
    questions: List[str]
    snippets: List[Snippet]
    analysis: Optional[str] = None


class Preprocessor:
    """Processes interview transcripts with AI and creates structured Interview objects."""
    def __init__(self):
        load_dotenv(override=True)
        self.openai_client: AzureOpenAI = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        )
        self.max_tokens: int = 7000  # Safety limit for tokens

    def _process_sentences(self, paragraph: str, current: dict, parts: list) -> dict:
        """Processes individual sentences of a paragraph.
        
        Args:
            paragraph: Paragraph to be processed
            current: Current chunk and token counter
            parts: List of completed parts
            
        Returns:
            Updated chunk and token counter
        """
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
            s_clean = f"{sentence} "
            s_tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(s_clean))
            if current["tokens"] + s_tokens > self.max_tokens:
                if current["chunk"]:
                    parts.append(current["chunk"])
                current = {"chunk": s_clean, "tokens": s_tokens}
            else:
                current["chunk"] += s_clean
                current["tokens"] += s_tokens
        return current

    def _split_transcript(self, text: str) -> List[str]:
        """
        Splits the transcript into smaller parts that are within the token limit.
        
        Args:
            text: The complete transcript
            
        Returns:
            List of transcript parts below the token limit
        """
        if len(tiktoken.encoding_for_model("gpt-4o").encode(text)) <= self.max_tokens:
            return [text]

        parts = []
        current = {"chunk": "", "tokens": 0}
        for paragraph in re.split(r"\n\s*\n", text):
            if not paragraph.strip():
                continue

            paragraph_clean = f"{paragraph.strip()}\n\n"
            p_tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(paragraph_clean))

            if p_tokens > self.max_tokens:
                current = self._process_sentences(paragraph, current, parts)
            elif current["tokens"] + p_tokens <= self.max_tokens:
                current["chunk"] += paragraph_clean
                current["tokens"] += p_tokens
            else:
                parts.append(current["chunk"])
                current = {"chunk": paragraph_clean, "tokens": p_tokens}

        if current["chunk"]:
            parts.append(current["chunk"])

        return parts

    def _merge_interview_parts(self, interview_parts: List[Interview]) -> Interview:
        """
        Merges multiple interview parts into a complete interview.
        Ensures that each speaker appears only once and all snippets
        are merged.

        Args:
            interview_parts: List of Interview objects

        Returns:
            A merged Interview object with all content
        """
        if len(interview_parts) == 1:
            return interview_parts[0]

        # Take the first interview as a base
        merged = Interview(transcript="", speakers=[], questions=[], snippets=[])

        # Collect all transcripts
        all_transcripts = [part.transcript for part in interview_parts]
        merged.transcript = max(all_transcripts, key=len) if all_transcripts else ""

        # Collect all unique speakers
        unique_speakers = set()
        for part in interview_parts:
            unique_speakers.update(part.speakers)
        merged.speakers = list(unique_speakers)

        # Collect all unique questions
        unique_questions = set()
        for part in interview_parts:
            unique_questions.update(part.questions)
        merged.questions = list(unique_questions)

        # Collect all snippets
        for part in interview_parts:
            merged.snippets.extend(part.snippets)

        return merged

    def process_part(self, part: str, i: int, transcript_parts: List[str]) -> Interview:
        """Processes a single part of the transcript and creates an Interview object.
        Args:
            part: Text part of the transcript
            i: Index of the current part
            transcript_parts: List of all transcript parts
            
        Returns:
            Interview object with extracted data
        """
        logging.info("Preprocessing chunk %d of %d from transcript", i+1, len(transcript_parts))
        return self.openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein hilfreicher Assistent, der Interview-"
                        "Transkripte vorverarbeitet.\n"
                        f"Dies ist Teil {i+1} von {len(transcript_parts)} des Transkripts.\n"
                        "Extrahiere alle Sprecher, Fragen und Gesprächsabschnitte.\n"
                        "Lasse das Analyse-Feld und das Transkript-Feld leer.\n"
                        "Jedes Statement soll ein extra Snippet sein mit dem jeweiligen Sprecher."
                    ),
                },
                {"role": "user", "content": part},
            ],
            response_format=Interview,
            max_tokens=16384
        )

    def _process_small_transcript(self, text: str) -> Interview:
        """Processes a small transcript directly."""
        response = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein hilfreicher Assistent, der Interview-"
                        "Transkripte vorverarbeitet. Lasse das Analyse-Feld vorerst leer."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format=Interview,
            max_tokens=16384,
        )
        return response.choices[0].message.parsed

    def ai_preprocess(
        self,
        text: str,
        augmented_questions: List[str] = None,
        use_multithreading: Optional[bool] = False,
        threads: Optional[int] = 1) -> Interview:
        """
        Processes a transcript with AI support and creates an Interview object.
        Splits large transcripts to avoid token limits.

        Args:
            text: The transcript to be processed
            augmented_questions: Optional list of questions
            use_multithreading: Flag whether to use multithreading
            threads: Number of threads to use

        Returns:
            An Interview object with structured data
        """
        if len(tiktoken.encoding_for_model("gpt-4o").encode(text)) <= self.max_tokens:
            result = self._process_small_transcript(text)
        else:
            # For large transcripts: Split and process separately
            parts = self._split_transcript(text)
            if use_multithreading and threads > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    part_indices = [(part, i) for i, part in enumerate(parts)]
                    results = executor.map(
                        lambda args: self.process_part(
                            args[0], args[1], parts
                        ).choices[0].message.parsed,
                        part_indices
                    )
                    result = self._merge_interview_parts(list(results))
            else:
                results = [
                    self.process_part(part, i, parts).choices[0].message.parsed
                    for i, part in enumerate(parts)
                ]
                result = self._merge_interview_parts(results)
            result.transcript = text
        if augmented_questions:
            result.questions = augmented_questions
        with open("preprocessor_debug.json", "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=4))
        return result
