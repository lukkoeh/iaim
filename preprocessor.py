"""Modul zur Vorverarbeitung von Interview-Transkripten mit KI-Unterstützung.
Dieses Modul verarbeitet große Transkripte in kleinere Teile und verarbeitet sie mit KI.
Die Ergebnisse werden in ein Interview-Objekt umgewandelt.
Das Modul wurde mithilfe von Cursor refactored, um ein 10/10 pylint Ergebnis zu erhalten.
Pre-Commit Hooks werden empfohlen.
"""

import os
import re
import concurrent.futures
from typing import Optional, List
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
import tiktoken


class Snippet(BaseModel):
    """Repräsentiert einen einzelnen Gesprächsabschnitt mit Sprecher und Text."""
    speaker: str
    text: str


class Interview(BaseModel):
    """Repräsentiert ein vollständiges Interview mit Transkript, Sprechern, Fragen und Snippets."""
    transcript: str
    speakers: List[str]
    questions: List[str]
    snippets: List[Snippet]
    analysis: Optional[str] = None


class Preprocessor:
    """Verarbeitet Interviewtranskripte mit KI und erstellt strukturierte Interview-Objekte."""
    def __init__(self):
        load_dotenv(override=True)
        self.openai_client: AzureOpenAI = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        )
        self.max_tokens: int = 7000  # Sicherheitsgrenze für Tokens

    def _process_sentences(self, paragraph: str, current: dict, parts: list) -> dict:
        """Verarbeitet einzelne Sätze eines Paragraphen.
        
        Args:
            paragraph: Zu verarbeitender Paragraph
            current: Aktueller Chunk und Token-Zähler
            parts: Liste der fertigen Teile
            
        Returns:
            Aktualisierter Chunk und Token-Zähler
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
        Teilt das Transkript in kleinere Teile auf, die innerhalb des Token-Limits liegen.
        
        Args:
            text: Das vollständige Transkript
            
        Returns:
            Liste von Transkript-Teilen unter dem Token-Limit
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
        Führt mehrere Interview-Teile zu einem vollständigen Interview zusammen.
        Stellt sicher, dass jeder Sprecher nur einmal vorkommt und alle Snippets
        zusammengeführt werden.

        Args:
            interview_parts: Liste von Interview-Objekten

        Returns:
            Ein zusammengeführtes Interview-Objekt mit allen Inhalten
        """
        if len(interview_parts) == 1:
            return interview_parts[0]

        # Nehme das erste Interview als Basis
        merged = Interview(transcript="", speakers=[], questions=[], snippets=[])

        # Sammle alle Transkripte
        all_transcripts = [part.transcript for part in interview_parts]
        merged.transcript = max(all_transcripts, key=len) if all_transcripts else ""

        # Sammle alle einzigartigen Sprecher
        unique_speakers = set()
        for part in interview_parts:
            unique_speakers.update(part.speakers)
        merged.speakers = list(unique_speakers)

        # Sammle alle einzigartigen Fragen
        unique_questions = set()
        for part in interview_parts:
            unique_questions.update(part.questions)
        merged.questions = list(unique_questions)

        # Sammle alle Snippets
        for part in interview_parts:
            merged.snippets.extend(part.snippets)

        return merged

    def process_part(self, part: str, i: int, transcript_parts: List[str]) -> Interview:
        """Verarbeitet einen einzelnen Teil des Transkripts und erstellt ein Interview-Objekt.
        Args:
            part: Textteil des Transkripts
            i: Index des aktuellen Teils
            transcript_parts: Liste aller Transkriptteile
            
        Returns:
            Interview-Objekt mit extrahierten Daten
        """
        print(f"[PREPROCESS] PROCESSING CHUNK {i+1} OF {len(transcript_parts)}")
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
        """Verarbeitet ein kleines Transkript direkt."""
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
        Verarbeitet ein Transkript mit KI-Unterstützung und erstellt ein Interview-Objekt.
        Teilt große Transkripte auf, um Token-Limits zu umgehen.

        Args:
            text: Das zu verarbeitende Transkript
            augmented_questions: Optionale Liste von Fragen
            use_multithreading: Flag, ob Multithreading verwendet werden soll
            threads: Anzahl der zu verwendenden Threads

        Returns:
            Ein Interview-Objekt mit strukturierten Daten
        """
        if len(tiktoken.encoding_for_model("gpt-4o").encode(text)) <= self.max_tokens:
            result = self._process_small_transcript(text)
        else:
            # Für große Transkripte: Aufteilung und separate Verarbeitung
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
