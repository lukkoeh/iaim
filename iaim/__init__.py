"""
IAIM (Interview Analysis and Information Mining) Package

This package combines transcript preprocessing and analysis capabilities,
providing tools to process interview transcripts, extract information,
and generate answers to questions using AI models.

Main components:
- Preprocessor: for processing and structuring raw interview transcripts
- TranscriptAnalyzer: for analyzing transcripts and generating answers
- Interview and Snippet: data models for structured interview data
"""

import logging
from typing import List, Optional

# Import all necessary classes and functions from modules
from .preprocessor import Preprocessor, Interview, Snippet
from .transcriptanalyzer import TranscriptAnalyzer
from .prompts import (
    SUMMARY_PROMPT,
    CLEANING_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    FULL_FORMAT_PROMPT,
    SPEAKERS_PROMPT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Export all important classes and functions at the package level
__all__ = [
    'Preprocessor',
    'TranscriptAnalyzer',
    'Interview',
    'Snippet',
    'analyze_transcript',
    'preprocess_transcript'
]

# Convenience functions to make common operations easier


def preprocess_transcript(
    text: str,
    augmented_questions: Optional[List[str]] = None,
    use_multithreading: bool = False,
    threads: int = 4
) -> Interview:
    """
    Preprocesses an interview transcript using AI to structure it.

    Args:
        text: The raw transcript text
        augmented_questions: Optional list of specific questions to include
        use_multithreading: Whether to use multiple threads for processing
        threads: Number of threads to use if multithreading is enabled

    Returns:
        An Interview object containing the structured data
    """
    preprocessor = Preprocessor()
    return preprocessor.ai_preprocess(
        text=text,
        augmented_questions=augmented_questions,
        use_multithreading=use_multithreading,
        threads=threads
    )


def analyze_transcript(
    questions_path: str,
    transcript_path: str,
    max_threads: int = 4,
    collection_name: str = "transcript_chunks"
) -> List[str]:
    """
    Performs complete analysis of a transcript file based on questions.

    Args:
        questions_path: Path to JSON file containing questions
        transcript_path: Path to text file containing the transcript
        max_threads: Maximum number of threads for parallel processing
        collection_name: Name for the ChromaDB collection

    Returns:
        List of question-answer pairs generated from the analysis
    """
    analyzer = TranscriptAnalyzer(
        max_threads=max_threads,
        collection_name=collection_name
    )
    return analyzer.analyze(
        questions_path=questions_path,
        transcript_path=transcript_path
    )


# This allows the package to be run as a script
if __name__ == "__main__":
    print("IAIM (Interview Analysis and Information Mining) Package")
    print("Use 'import iaim' to access the package functionality.")
