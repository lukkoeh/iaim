"""
Module for analyzing transcripts using AI models.

This module contains the TranscriptAnalyzer class, which processes transcripts,
extracts information, and generates answers to questions. It uses Azure OpenAI
and ChromaDB for processing and storing data.
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
from static.prompts import (
    SUMMARY_PROMPT,
    CLEANING_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    FULL_FORMAT_PROMPT
)
from preprocessor import (
    Preprocessor,
    Interview
)

# pylint: disable-next=too-many-instance-attributes
class TranscriptAnalyzer:
    """
    Class for analyzing transcripts using AI models.

    This class enables the processing of transcripts, extraction of information
    based on questions, and generation of answers.
    """

    def __init__(
        self,
        max_threads: int = 4,
        collection_name: str = "transcript_chunks",
    ):
        """
        Initializes the TranscriptAnalyzer.

        Args:
            max_threads: Maximum number of threads for multithreading operations
            collection_name: Name of the collection in ChromaDB
        """
        self.max_threads: int = max_threads
        self.openai_client: Optional[AzureOpenAI] = None
        self.db_client: Optional[chromadb.Client] = None
        self.collection: Optional[Any] = None
        self.collection_name: str = collection_name
        self.preprocessor: Preprocessor = Preprocessor()
        self.interview: Optional[Interview] = None
        self.questions_df: Optional[pd.DataFrame] = None
        # Reduce the number of instance attributes by using a dictionary
        self.data: Dict[str, Any] = {
            "clean_chunks": [],
            "extracted_information": [],
            "final_answers": [],
            "question_answer_pairs": []
        }

        # Load environment variables
        load_dotenv(override=True)

    def initialize_clients(self) -> None:
        """
        Initializes the OpenAI and ChromaDB clients.
        """
        # Initialize the Azure OpenAI Client
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        )
        logging.info("Azure OpenAI Client initialized")

        # Configure the Embedding function
        # Import the class directly to fix the no-member problem
        # pylint: disable-next=no-member
        default_ef = embed_fns.OpenAIEmbeddingFunction(
            api_key=os.getenv("AZURE_EMBEDDING_KEY"),
            api_base=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_type="azure",
            api_version="2023-05-15",
            model_name="text-embedding-3-large",
        )
        logging.info("Embedding function configured")

        # Create ChromaDB Client
        self.db_client = chromadb.Client()
        logging.info("ChromaDB Client created")

        # Create a collection in ChromaDB
        logging.info(
            "Deleting existing collection '%s' if present...",
            self.collection_name
        )
        try:
            self.db_client.delete_collection(name=self.collection_name)
            logging.info("Existing collection deleted")
        except ValueError as e:
            # More specific exception instead of general Exception
            logging.info(
                "No existing collection found or error while deleting: %s",
                e
            )
        except chroma_errors.NotFoundError as e:
            if "Collection not found" in str(e):
                logging.info(
                    "Collection not found: %s",
                    e
                )

        self.collection = self.db_client.create_collection(
            name=self.collection_name, embedding_function=default_ef
        )
        logging.info(
            "New ChromaDB collection '%s' created",
            self.collection_name
        )

    def load_data(self, questions_path: str, transcript_path: str) -> None:
        """
        Loads questions and transcript from files.

        Args:
            questions_path: Path to JSON file with questions
            transcript_path: Path to text file with the transcript
        """
        # Load questions from JSON file
        with open(questions_path, "r", encoding="utf-8") as f:
            json_data: Dict = json.load(f)
        self.questions_df = pd.DataFrame(json_data)
        logging.info(
            "Questions loaded: %d entries found",
            len(self.questions_df)
        )

        # Load transcript from text file
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript: str = f.read()
        logging.info("Transcript loaded: %d characters", len(transcript))

        # Preprocess the transcript
        logging.info("%s", "\n" + "-" * 50)
        logging.info(
            "Starting transcript preprocessing... (With Preprocessor)")
        self.interview = self.preprocessor.ai_preprocess(
            transcript,
            augmented_questions=self.questions_df["question"].tolist(),
            use_multithreading=True,
            threads=self.max_threads,
        )
        logging.info("Transcript preprocessing completed")

        # Show the preprocessed transcript for verification
        logging.info("\nPreprocessed transcript:")
        # Only the first 200 characters
        logging.info("%s...", self.interview.transcript[:200])
        logging.info("%s", "-" * 50)

    def _process_chunk(self, args: Tuple) -> Dict:
        """
        Processes a single chunk in multithreading mode.

        Args:
            args: Tuple with (chunk, index, total_chunks, openai_client)

        Returns:
            Processed chunk with summary
        """
        chunk, i, total_chunks, openai_client = args
        logging.info(
            "Cleaning chunk %d/%d and summarizing it (1st summary)",
            i + 1, total_chunks
        )
        text = chunk.text
        speaker = chunk.speaker

        # Remove speaker marking from text
        clean_text = text

        # Clean the text using OpenAI to remove unnecessary characters and filler words
        clean_text = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CLEANING_PROMPT},
                {"role": "user", "content": clean_text},
            ],
        )
        clean_text = clean_text.choices[0].message.content

        # Generate an initial summary of each chunk and append it to the text with \n\n
        summary = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": clean_text},
            ],
        )
        summary = summary.choices[0].message.content
        clean_text = f"{clean_text}\n\n{summary}"

        return {"text": clean_text, "speaker": speaker, "index": i}

    def process_chunks(self) -> None:
        """
        Processes all chunks of the transcript.
        """
        logging.info("\nCleaning chunks and extracting speakers...")
        self.data["clean_chunks"] = []
        speaker_stats = {}

        # Multithreaded chunk processing
        chunk_results = []

        if self.max_threads > 1:
            logging.info(
                "Using multithreading with maximum %d threads for chunk processing",
                self.max_threads
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_threads
            ) as executor:
                # Create a list of arguments for each chunk
                chunk_args = [
                    (chunk, i, len(self.interview.snippets), self.openai_client)
                    for i, chunk in enumerate(self.interview.snippets)
                ]
                # Process all chunks in parallel
                chunk_results = list(
                    executor.map(
                        self._process_chunk,
                        chunk_args))
        else:
            logging.info("Using sequential processing for chunks")
            for i, chunk in enumerate(self.interview.snippets):
                result = self._process_chunk(
                    (chunk, i, len(self.interview.snippets), self.openai_client)
                )
                chunk_results.append(result)

        # Sort results by original index
        chunk_results.sort(key=lambda x: x["index"])

        # Process the results
        for result in chunk_results:
            clean_text = result["text"]
            speaker = result["speaker"]
            i = result["index"]

            # Count speakers for statistics
            if speaker in speaker_stats:
                speaker_stats[speaker] += 1
            else:
                speaker_stats[speaker] = 1

            if clean_text:  # Ignore empty chunks
                self.data["clean_chunks"].append(clean_text)
                if (
                    i < 3 or i >= len(self.interview.snippets) - 3
                ):  # Show the first and last 3 chunks
                    logging.info(
                        "Chunk %d (Speaker: %s): %s...",
                        i + 1, speaker, clean_text[:50]
                    )

        logging.info("\nCleaned chunks: %d", len(self.data["clean_chunks"]))
        logging.info("Speaker statistics:")
        for speaker, count in speaker_stats.items():
            logging.info("  %s: %d chunks", speaker, count)

        # Add chunks to ChromaDB
        logging.info("%s", "\n" + "-" * 50)
        logging.info(
            "Adding %d chunks to ChromaDB...",
            len(self.data["clean_chunks"])
        )
        self.collection.add(
            ids=[str(i) for i in range(len(self.data["clean_chunks"]))],
            documents=self.data["clean_chunks"],
        )
        logging.info("Chunks successfully added to ChromaDB")
        logging.info(
            "Number of documents in collection: %d",
            self.collection.count()
        )

    def _process_question(self, args: Tuple) -> Dict:
        """
        Processes a single question in multithreading mode.

        Args:
            args: Tuple with (question, question_index, total_questions, collection, openai_client)

        Returns:
            Extracted information for the question
        """
        question, q, total_questions, collection, openai_client = args
        logging.info("Question: %s (Question %d/%d)", question, q, total_questions)

        # Get relevant chunks for the question
        relevant_chunks = collection.query(query_texts=[question], n_results=5)

        # Use relevant chunks for extraction
        chunk_texts = relevant_chunks["documents"][0]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(
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
        Extracts information from chunks based on questions.
        """
        self.data["extracted_information"] = []

        # Multithreaded question processing
        if self.max_threads > 1:
            logging.info(
                "Using multithreading with %d threads for question processing",
                self.max_threads
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_threads
            ) as executor:
                # Create a list of arguments for each question
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
                # Process all questions in parallel
                self.data["extracted_information"] = list(
                    executor.map(self._process_question, question_args)
                )
        else:
            logging.info("Using sequential processing for questions")
            # Iterate through questions and chunks and perform RAG-based
            # information extraction
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

        # Save extracted information to a JSON file
        with open("extracted_information.json", "w", encoding="utf-8") as f:
            json.dump(self.data["extracted_information"], f, ensure_ascii=False)
            logging.info(
                "Extracted information saved in 'extracted_information.json'"
            )

    def _process_final_answer(self, args: Tuple) -> Dict:
        """
        Processes the final answer for a question in multithreading mode.

        Args:
            args: Tuple with (question, question_index, total_questions, 
                  question_info, openai_client)

        Returns:
            Final answer for the question
        """
        question, q, total_questions, question_info, openai_client = args
        logging.info("Processing question %d/%d: %s", q, total_questions, question)

        # Prepare content for this question
        question_content = (
            f"Frage: {question_info['question']}\n\n"
            f"Interpretation: {question_info['interpretation']}"
        )

        # Generate answer for this single question
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": FULL_FORMAT_PROMPT.format(interview_question=question),
                },
                {"role": "user", "content": question_content},
            ],
            max_tokens=16384,
            temperature=1,
        )

        question_answer = response.choices[0].message.content

        # Continue generation if answer is not complete
        messages_history = [
            {
                "role": "system",
                "content": FULL_FORMAT_PROMPT.format(interview_question=question),
            },
            {"role": "user", "content": question_content},
            {"role": "assistant", "content": question_answer},
        ]

        while response.choices[0].finish_reason != "stop":
            logging.info(
                "Answer for question %d is not complete yet, continuing...",
                q
            )
            messages_history.append(
                {"role": "user", "content": "Please continue your analysis..."}
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
        logging.info("Question %d processed. Tokens: %d", q, tokens)

        return {"question": question, "answer": question_answer}

    def generate_answers(self) -> None:
        """
        Generates answers based on extracted information.
        """
        # Combine extracted information into a string
        combined_information = "\n\n".join(
            [
                f"Frage: {item['question']}\n\nInterpretation: {item['interpretation']}"
                for item in self.data["extracted_information"]
            ]
        )
        tokens = len(tiktoken.encoding_for_model(
            "gpt-4o").encode(combined_information))
        logging.info(
            "Number of tokens in combined information: %d",
            tokens
        )

        if tokens < 125000:
            logging.info(
                "The combined information fits within gpt-4o's context window"
            )
            # Process each question individually
            self.data["final_answers"] = []

            # Multithreaded final answer processing
            if self.max_threads > 1:
                logging.info(
                    "Using multithreading with %d threads for final answers",
                    self.max_threads
                )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_threads
                ) as executor:
                    # Create a list of arguments for each question
                    final_args = []
                    for q, question in enumerate(
                            self.questions_df["question"], 1):
                        # Filter extracted information for this specific question
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

                    # Process all final answers in parallel
                    if final_args:
                        self.data["final_answers"] = list(
                            executor.map(
                                self._process_final_answer, final_args)
                        )
            else:
                logging.info(
                    "Using sequential processing for final answers")
                for q, question in enumerate(self.questions_df["question"], 1):
                    # Filter extracted information for this specific question
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

            # Create question-answer pairs
            self.data["question_answer_pairs"] = [
                f"Frage: {item['question']}\n\nAntwort: {item['answer']}"
                for item in self.data["final_answers"]
            ]

            # Combine all answers into a complete document
            final_answer = "# Complete Interview Analysis\n\n"
            final_answer += (
                "# Introduction\n\n"
                "This document contains the analysis of an interview "
                "based on the questions asked.\n\n"
            )
            final_answer += "# Main Section\n\n"
            for item in self.data["final_answers"]:
                # Remove any existing introductions and
                # conclusions from individual answers
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

                # Add the cleaned answer to the complete document
                final_answer += answer_content + "\n\n"

            final_answer += (
                "# Conclusion\n\n"
                "The analysis above summarizes the key findings "
                "from the interview."
            )

            # Save the complete answer
            with open("final_answer.txt", "w", encoding="utf-8") as f:
                f.write(final_answer)

            tokens = len(tiktoken.encoding_for_model(
                "gpt-4o").encode(final_answer))
            logging.info(
                "Number of tokens in final complete answer: %d",
                tokens
            )
            logging.info(
                "Final complete answer saved in 'final_answer.txt'")
        else:
            logging.info(
                "The combined information does not fit within gpt-4o's context window"
            )
            # Implement an alternative for large content
            logging.info("Processing answers in batches...")
            # Code for batch processing could follow here
            self._process_in_batches()

    def _process_in_batches(self) -> None:
        """
        Processes answers in batches when combined information is too large.
        """
        # Implementation for batch processing
        logging.error("Batch processing not yet implemented")

    def analyze(self, questions_path: str, transcript_path: str) -> List[str]:
        """
        Performs complete analysis of the transcript.

        Args:
            questions_path: Path to JSON file with questions
            transcript_path: Path to text file with the transcript

        Returns:
            List of question-answer pairs
        """
        logging.info("=" * 50)
        logging.info("Starting transcript analysis")
        logging.info("=" * 50)

        # Initialize clients
        self.initialize_clients()

        # Load data
        self.load_data(questions_path, transcript_path)

        # Process chunks
        self.process_chunks()

        # Extract information
        self.extract_information()

        # Generate answers
        self.generate_answers()

        logging.info("=" * 50)
        logging.info("Transcript analysis completed")
        logging.info("=" * 50)

        return self.data["question_answer_pairs"]


if __name__ == "__main__":
    # Example of using the class
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    analyzer = TranscriptAnalyzer(max_threads=4)
    qa_pairs = analyzer.analyze(
        questions_path="data/questions.json",
        transcript_path="data/sample_transcript.txt",
    )

    logging.info("Number of question-answer pairs: %d", len(qa_pairs))
    if qa_pairs:
        logging.info("Example of a question-answer pair:")
        # Show only the first 200 characters
        logging.info("%s...", qa_pairs[0][:200])
