"""
Modul zur Analyse von Transkripten mit Hilfe von KI-Modellen.

Dieses Modul enthält die TranscriptAnalyzer-Klasse, die Transkripte verarbeitet,
Informationen extrahiert und Antworten auf Fragen generiert. Es nutzt Azure OpenAI
und ChromaDB für die Verarbeitung und Speicherung der Daten.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
import concurrent.futures
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import tiktoken
import chromadb
# pylint: disable-next=no-member
import chromadb.utils.embedding_functions as embed_fns
import chromadb.errors as chroma_errors
from prompts import (
    summary_prompt,
    cleaning_prompt,
    extraction_system_prompt,
    extraction_prompt,
    full_format_prompt,
)
from preprocessor import (
    Preprocessor,
    Interview
)

# pylint: disable-next=too-many-instance-attributes
class TranscriptAnalyzer:
    """
    Klasse zur Analyse von Transkripten mit Hilfe von KI-Modellen.

    Diese Klasse ermöglicht die Verarbeitung von Transkripten, die Extraktion von Informationen
    basierend auf Fragen und die Generierung von Antworten.
    """

    def __init__(
        self,
        max_threads: int = 4,
        db_path: str = "chroma_db",
        collection_name: str = "transcript_chunks",
    ):
        """
        Initialisiert den TranscriptAnalyzer.

        Args:
            max_threads: Maximale Anzahl der Threads für Multithreading-Operationen
            db_path: Pfad zur ChromaDB
            collection_name: Name der Kollektion in der ChromaDB
        """
        self.max_threads: int = max_threads
        self.openai_client: Optional[AzureOpenAI] = None
        self.db_client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Any] = None
        self.collection_name: str = collection_name
        self.db_path: str = db_path
        self.preprocessor: Preprocessor = Preprocessor()
        self.interview: Optional[Interview] = None
        self.questions_df: Optional[pd.DataFrame] = None
        # Reduziere die Anzahl der Instanzattribute durch Verwendung eines Dictionaries
        self.data: Dict[str, Any] = {
            "clean_chunks": [],
            "extracted_information": [],
            "final_answers": [],
            "question_answer_pairs": []
        }

        # Lade Umgebungsvariablen
        load_dotenv(override=True)

    def initialize_clients(self) -> None:
        """
        Initialisiert die OpenAI- und ChromaDB-Clients.
        """
        # Initialisiere den Azure OpenAI Client
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        )
        logging.info("🤖 Azure OpenAI Client initialisiert")

        # Konfiguriere die Embedding-Funktion
        # Importiere die Klasse direkt, um das no-member Problem zu beheben
        # pylint: disable-next=no-member
        default_ef = embed_fns.OpenAIEmbeddingFunction(
            api_key=os.getenv("AZURE_EMBEDDING_KEY"),
            api_base=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_type="azure",
            api_version="2023-05-15",
            model_name="text-embedding-3-large",
        )
        logging.info("🧠 Embedding-Funktion konfiguriert")

        # Erstelle ChromaDB Client
        self.db_client = chromadb.PersistentClient(path=self.db_path)
        logging.info("💾 ChromaDB Client erstellt mit Pfad: %s", self.db_path)

        # Erstelle eine Kollektion in der ChromaDB
        logging.info(
            "🗑️ Lösche vorhandene Kollektion '%s' falls vorhanden...",
            self.collection_name
        )
        try:
            self.db_client.delete_collection(name=self.collection_name)
            logging.info("✅ Vorhandene Kollektion gelöscht")
        except ValueError as e:
            # Spezifischere Ausnahme anstelle von allgemeinem Exception
            logging.info(
                "ℹ️ Keine vorhandene Kollektion gefunden oder Fehler beim Löschen: %s",
                e
            )
        except chroma_errors.NotFoundError as e:
            if "Collection not found" in str(e):
                logging.info(
                    "ℹ️ Kollektion nicht gefunden: %s",
                    e
                )

        self.collection = self.db_client.create_collection(
            name=self.collection_name, embedding_function=default_ef
        )
        logging.info(
            "📦 Neue ChromaDB Kollektion '%s' erstellt",
            self.collection_name
        )

    def load_data(self, questions_path: str, transcript_path: str) -> None:
        """
        Lädt Fragen und Transkript aus Dateien.

        Args:
            questions_path: Pfad zur JSON-Datei mit Fragen
            transcript_path: Pfad zur Textdatei mit dem Transkript
        """
        # Lade Fragen aus JSON-Datei
        with open(questions_path, "r", encoding="utf-8") as f:
            json_data: Dict = json.load(f)
        self.questions_df = pd.DataFrame(json_data)
        logging.info(
            "❓ Fragen geladen: %d Einträge gefunden",
            len(self.questions_df)
        )

        # Lade Transkript aus Textdatei
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript: str = f.read()
        logging.info("📄 Transkript geladen: %d Zeichen", len(transcript))

        # Vorverarbeite das Transkript
        logging.info("%s", "\n" + "-" * 50)
        logging.info(
            "🔄 Starte Transkriptvorverarbeitung... (Mit Preprocessor)")
        self.interview = self.preprocessor.ai_preprocess(
            transcript,
            augmented_questions=self.questions_df["question"].tolist(),
            use_multithreading=True,
            threads=self.max_threads,
        )
        logging.info("✅ Transkriptvorverarbeitung abgeschlossen")

        # Zeige das vorverarbeitete Transkript zur Überprüfung
        logging.info("\n📝 Vorverarbeitetes Transkript:")
        # Nur die ersten 200 Zeichen
        logging.info("%s...", self.interview.transcript[:200])
        logging.info("%s", "-" * 50)

    def _process_chunk(self, args: Tuple) -> Dict:
        """
        Verarbeitet einen einzelnen Chunk im Multithreading-Modus.

        Args:
            args: Tuple mit (chunk, index, total_chunks, openai_client)

        Returns:
            Verarbeiteter Chunk mit Zusammenfassung
        """
        chunk, i, total_chunks, openai_client = args
        logging.info(
            "🔄 Bereinige Chunk %d/%d und fasse ihn zusammen (1. Zusammenfassung)",
            i + 1, total_chunks
        )
        text = chunk.text
        speaker = chunk.speaker

        # Entferne die Sprechermarkierung aus dem Text
        clean_text = text

        # Bereinige den Text mittels OpenAI um unnötige Zeichen und Füllwörter
        clean_text = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cleaning_prompt},
                {"role": "user", "content": clean_text},
            ],
        )
        clean_text = clean_text.choices[0].message.content

        # Erzeuge eine erste Zusammenfassung jedes Chunks und füge es einfach
        # an den Text mit \n\n zusammen
        summary = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": clean_text},
            ],
        )
        summary = summary.choices[0].message.content
        clean_text = f"{clean_text}\n\n{summary}"

        return {"text": clean_text, "speaker": speaker, "index": i}

    def process_chunks(self) -> None:
        """
        Verarbeitet alle Chunks des Transkripts.
        """
        logging.info("\n🧹 Bereinige Chunks und extrahiere Sprecher...")
        self.data["clean_chunks"] = []
        speaker_stats = {}

        # Multithreaded Chunk-Verarbeitung
        chunk_results = []

        if self.max_threads > 1:
            logging.info(
                "🧵 Verwende Multithreading mit maximal %d Threads für die Chunk-Verarbeitung",
                self.max_threads
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_threads
            ) as executor:
                # Erstelle eine Liste von Argumenten für jeden Chunk
                chunk_args = [
                    (chunk, i, len(self.interview.snippets), self.openai_client)
                    for i, chunk in enumerate(self.interview.snippets)
                ]
                # Verarbeite alle Chunks parallel
                chunk_results = list(
                    executor.map(
                        self._process_chunk,
                        chunk_args))
        else:
            logging.info("🔄 Verwende sequentielle Verarbeitung für Chunks")
            for i, chunk in enumerate(self.interview.snippets):
                result = self._process_chunk(
                    (chunk, i, len(self.interview.snippets), self.openai_client)
                )
                chunk_results.append(result)

        # Sortiere die Ergebnisse nach dem ursprünglichen Index
        chunk_results.sort(key=lambda x: x["index"])

        # Verarbeite die Ergebnisse
        for result in chunk_results:
            clean_text = result["text"]
            speaker = result["speaker"]
            i = result["index"]

            # Zähle Sprecher für Statistik
            if speaker in speaker_stats:
                speaker_stats[speaker] += 1
            else:
                speaker_stats[speaker] = 1

            if clean_text:  # Ignoriere leere Chunks
                self.data["clean_chunks"].append(clean_text)
                if (
                    i < 3 or i >= len(self.interview.snippets) - 3
                ):  # Zeige die ersten und letzten 3 Chunks
                    logging.info(
                        "📄 Chunk %d (Sprecher: %s): %s...",
                        i + 1, speaker, clean_text[:50]
                    )

        logging.info("\n✅ Bereinigte Chunks: %d", len(self.data["clean_chunks"]))
        logging.info("👥 Sprecher-Statistik:")
        for speaker, count in speaker_stats.items():
            logging.info("  👤 %s: %d Chunks", speaker, count)

        # Füge die Chunks zur ChromaDB hinzu
        logging.info("%s", "\n" + "-" * 50)
        logging.info(
            "💾 Füge %d Chunks zur ChromaDB hinzu...",
            len(self.data["clean_chunks"])
        )
        self.collection.add(
            ids=[str(i) for i in range(len(self.data["clean_chunks"]))],
            documents=self.data["clean_chunks"],
        )
        logging.info("✅ Chunks erfolgreich zur ChromaDB hinzugefügt")
        logging.info(
            "📊 Anzahl der Dokumente in der Kollektion: %d",
            self.collection.count()
        )

    def _process_question(self, args: Tuple) -> Dict:
        """
        Verarbeitet eine einzelne Frage im Multithreading-Modus.

        Args:
            args: Tuple mit (question, question_index, total_questions, collection, openai_client)

        Returns:
            Extrahierte Informationen für die Frage
        """
        question, q, total_questions, collection, openai_client = args
        logging.info("❓ Frage: %s (Frage %d/%d)", question, q, total_questions)

        # Hole die relevanten Chunks für die Frage
        relevant_chunks = collection.query(query_texts=[question], n_results=5)

        # Verwende die relevanten Chunks für die Extraktion
        chunk_texts = relevant_chunks["documents"][0]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": extraction_system_prompt},
                {
                    "role": "user",
                    "content": extraction_prompt.format(
                        interview_chunk="\n".join(chunk_texts),
                        interview_question=question,
                        relevant_chunks=chunk_texts,
                    ),
                },
            ],
        )
        response = response.choices[0].message.content
        return {"question": question, "interpretation": response}

    def extract_information(self) -> None:
        """
        Extrahiert Informationen aus den Chunks basierend auf den Fragen.
        """
        self.data["extracted_information"] = []

        # Multithreaded Fragen-Verarbeitung
        if self.max_threads > 1:
            logging.info(
                "🧵 Verwende Multithreading mit %d Threads für die Fragen-Verarbeitung",
                self.max_threads
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_threads
            ) as executor:
                # Erstelle eine Liste von Argumenten für jede Frage
                question_args = [
                    (
                        question,
                        q,
                        len(self.questions_df),
                        self.collection,
                        self.openai_client,
                    )
                    for q, question in enumerate(self.questions_df["question"], 1)
                ]
                # Verarbeite alle Fragen parallel
                self.data["extracted_information"] = list(
                    executor.map(self._process_question, question_args)
                )
        else:
            logging.info("🔄 Verwende sequentielle Verarbeitung für Fragen")
            # Durchlaufe die Fragen und die Chunks und führe eine RAG-basierte
            # Extraktion von Informationen durch
            for q, question in enumerate(self.questions_df["question"], 1):
                result = self._process_question(
                    (
                        question,
                        q,
                        len(self.questions_df),
                        self.collection,
                        self.openai_client,
                    )
                )
                self.data["extracted_information"].append(result)

        # Speichere die extrahierten Informationen in einer JSON-Datei
        with open("extracted_information.json", "w", encoding="utf-8") as f:
            json.dump(self.data["extracted_information"], f, ensure_ascii=False)
            logging.info(
                "💾 Extrahierte Informationen in 'extracted_information.json' gespeichert"
            )

    def _process_final_answer(self, args: Tuple) -> Dict:
        """
        Verarbeitet die finale Antwort für eine Frage im Multithreading-Modus.

        Args:
            args: Tuple mit (question, question_index, total_questions, 
                  question_info, openai_client)

        Returns:
            Finale Antwort für die Frage
        """
        question, q, total_questions, question_info, openai_client = args
        logging.info("🔄 Verarbeite Frage %d/%d: %s", q, total_questions, question)

        # Bereite den Inhalt für diese Frage vor
        question_content = (
            f"Frage: {question_info['question']}\n\n"
            f"Interpretation: {question_info['interpretation']}"
        )

        # Generiere die Antwort für diese einzelne Frage
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": full_format_prompt.format(interview_question=question),
                },
                {"role": "user", "content": question_content},
            ],
            max_tokens=16384,
            temperature=1,
        )

        question_answer = response.choices[0].message.content

        # Setze die Generierung fort, falls die Antwort nicht vollständig ist
        messages_history = [
            {
                "role": "system",
                "content": full_format_prompt.format(interview_question=question),
            },
            {"role": "user", "content": question_content},
            {"role": "assistant", "content": question_answer},
        ]

        while response.choices[0].finish_reason != "stop":
            logging.info(
                "🔄 Die Antwort für Frage %d ist noch nicht vollständig, setze fort...",
                q
            )
            messages_history.append(
                {"role": "user", "content": "Bitte setze deine Analyse fort..."}
            )

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages_history,
                max_tokens=16384,
                temperature=1,
            )

            continuation = response.choices[0].message.content
            question_answer += "\n" + continuation
            messages_history.append(
                {"role": "assistant", "content": continuation})

        tokens = len(tiktoken.encoding_for_model(
            "gpt-4o").encode(question_answer))
        logging.info("✅ Frage %d verarbeitet. Tokens: %d", q, tokens)

        return {"question": question, "answer": question_answer}

    def generate_answers(self) -> None:
        """
        Generiert Antworten basierend auf den extrahierten Informationen.
        """
        # Kombiniere die extrahierten Informationen zu einem String
        combined_information = "\n\n".join(
            [
                f"Frage: {item['question']}\n\nInterpretation: {item['interpretation']}"
                for item in self.data["extracted_information"]
            ]
        )
        tokens = len(tiktoken.encoding_for_model(
            "gpt-4o").encode(combined_information))
        logging.info(
            "🔢 Anzahl der Tokens in der kombinierten Information: %d",
            tokens
        )

        if tokens < 125000:
            logging.info(
                "✅ Die kombinierte Information passt in das Kontextfenster von gpt-4o"
            )
            # Verarbeite jede Frage einzeln
            self.data["final_answers"] = []

            # Multithreaded finale Antworten-Verarbeitung
            if self.max_threads > 1:
                logging.info(
                    "🧵 Verwende Multithreading mit %d Threads für die finalen Antworten",
                    self.max_threads
                )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_threads
                ) as executor:
                    # Erstelle eine Liste von Argumenten für jede Frage
                    final_args = []
                    for q, question in enumerate(
                            self.questions_df["question"], 1):
                        # Filtere die extrahierten Informationen für diese
                        # spezifische Frage
                        question_info = next(
                            (
                                item
                                for item in self.data["extracted_information"]
                                if item["question"] == question
                            ),
                            None,
                        )
                        if question_info:
                            final_args.append(
                                (
                                    question,
                                    q,
                                    len(self.questions_df),
                                    question_info,
                                    self.openai_client,
                                )
                            )

                    # Verarbeite alle finalen Antworten parallel
                    if final_args:
                        self.data["final_answers"] = list(
                            executor.map(
                                self._process_final_answer, final_args)
                        )
            else:
                logging.info(
                    "🔄 Verwende sequentielle Verarbeitung für finale Antworten")
                for q, question in enumerate(self.questions_df["question"], 1):
                    # Filtere die extrahierten Informationen für diese
                    # spezifische Frage
                    question_info = next(
                        (
                            item
                            for item in self.data["extracted_information"]
                            if item["question"] == question
                        ),
                        None,
                    )

                    if question_info:
                        result = self._process_final_answer(
                            (
                                question,
                                q,
                                len(self.questions_df),
                                question_info,
                                self.openai_client,
                            )
                        )
                        self.data["final_answers"].append(result)

            # Erstelle die Frage-Antwort-Paare
            self.data["question_answer_pairs"] = [
                f"Frage: {item['question']}\n\nAntwort: {item['answer']}"
                for item in self.data["final_answers"]
            ]

            # Kombiniere alle Antworten zu einem Gesamtdokument
            final_answer = "# Gesamtauswertung des Interviews\n\n"
            final_answer += (
                "# Einleitung\n\n"
                "Dieses Dokument enthält die Auswertung eines Interviews "
                "basierend auf den gestellten Fragen.\n\n"
            )
            final_answer += "# Hauptteil\n\n"
            for item in self.data["final_answers"]:
                # Entferne eventuell vorhandene Einleitungen und
                # Schlussfolgerungen aus den Einzelantworten
                answer_content = item["answer"]
                answer_content = re.sub(
                    r"# Einleitung.*?(?=## Frage|# Hauptteil)",
                    "",
                    answer_content,
                    flags=re.DOTALL,
                )
                answer_content = re.sub(
                    r"# Schlussfolgerung.*", "", answer_content, flags=re.DOTALL
                )

                # Füge die bereinigte Antwort zum Gesamtdokument hinzu
                final_answer += answer_content + "\n\n"

            final_answer += (
                "# Schlussfolgerung\n\n"
                "Die obige Analyse fasst die wichtigsten Erkenntnisse "
                "aus dem Interview zusammen."
            )

            # Speichere die Gesamtantwort
            with open("final_answer.txt", "w", encoding="utf-8") as f:
                f.write(final_answer)

            tokens = len(tiktoken.encoding_for_model(
                "gpt-4o").encode(final_answer))
            logging.info(
                "🔢 Anzahl der Tokens in der finalen Gesamtantwort: %d",
                tokens
            )
            logging.info(
                "📄 Finale Gesamtantwort in 'final_answer.txt' gespeichert")
        else:
            logging.info(
                "⚠️ Die kombinierte Information passt nicht in den Kontextfenster von gpt-4o"
            )
            # Implementiere eine Alternative für große Inhalte
            logging.info("🔄 Verarbeite Antworten in Batches...")
            # Hier könnte Code für eine Batch-Verarbeitung folgen
            self._process_in_batches()

    def _process_in_batches(self) -> None:
        """
        Verarbeitet die Antworten in Batches, wenn die kombinierte Information zu groß ist.
        """
        # Implementierung für Batch-Verarbeitung
        logging.error("⚠️ Batch-Verarbeitung noch nicht implementiert")

    def analyze(self, questions_path: str, transcript_path: str) -> List[str]:
        """
        Führt die vollständige Analyse des Transkripts durch.

        Args:
            questions_path: Pfad zur JSON-Datei mit Fragen
            transcript_path: Pfad zur Textdatei mit dem Transkript

        Returns:
            Liste der Frage-Antwort-Paare
        """
        logging.info("=" * 50)
        logging.info("🚀 Starte Transkriptanalyse")
        logging.info("=" * 50)

        # Initialisiere Clients
        self.initialize_clients()

        # Lade Daten
        self.load_data(questions_path, transcript_path)

        # Verarbeite Chunks
        self.process_chunks()

        # Extrahiere Informationen
        self.extract_information()

        # Generiere Antworten
        self.generate_answers()

        logging.info("=" * 50)
        logging.info("🎉 Transkriptanalyse abgeschlossen")
        logging.info("=" * 50)

        return self.data["question_answer_pairs"]


if __name__ == "__main__":
    # Beispiel für die Verwendung der Klasse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    analyzer = TranscriptAnalyzer(max_threads=4)
    qa_pairs = analyzer.analyze(
        questions_path="data/questions.json",
        transcript_path="data/sample_transcript.txt",
    )

    logging.info("📊 Anzahl der Frage-Antwort-Paare: %d", len(qa_pairs))
    if qa_pairs:
        logging.info("📝 Beispiel für ein Frage-Antwort-Paar:")
        # Zeige nur die ersten 200 Zeichen
        logging.info("%s...", qa_pairs[0][:200])
