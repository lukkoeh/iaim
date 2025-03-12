# IAIM - Interview Analysis and Information Mining

A Python toolkit for transforming interview transcripts into structured knowledge for Retrieval-Augmented Generation (RAG) systems.

## Overview

IAIM processes interview transcripts using AI to extract meaningful information, summarize content, and prepare data for knowledge retrieval systems. The toolkit leverages Azure OpenAI services and ChromaDB for vector storage.

## Features

- **Transcript Preprocessing**: Clean and structure raw interview transcripts
- **Speaker Recognition**: Automatically identify and tag different speakers
- **Question Extraction**: Identify and categorize questions within interviews
- **AI-Powered Summarization**: Generate concise summaries of interview segments
- **Information Extraction**: Extract key information based on predefined questions
- **Vector Storage**: Store processed data in ChromaDB for efficient retrieval
- **Multithreading Support**: Process large transcripts efficiently with parallel processing

## Prerequisites

- Python 3.13+
- Poetry for dependency management
- Azure OpenAI API access
- Docker and Docker Compose (optional, for Neo4j integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iaim.git
   cd iaim
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_API_KEY="your-api-key"
   AZURE_API_VERSION="your-api-version"
   AZURE_ENDPOINT="your-azure-endpoint"
   AZURE_EMBEDDING_KEY="your-embedding-key"
   AZURE_EMBEDDING_ENDPOINT="your-embedding-endpoint"
   AZURE_EMBEDDING_MODEL="your-embedding-model"
   ```

## Usage Examples

### Preprocessing an Interview Transcript

```python
from modules.preprocessor import Preprocessor

# Initialize the preprocessor
preprocessor = Preprocessor()

# Process a transcript file
with open("samples/interview.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

# Process the transcript with AI assistance
interview = preprocessor.ai_preprocess(
    text=transcript,
    use_multithreading=True,
    threads=4
)

# Access the structured data
print(f"Speakers: {interview.speakers}")
print(f"Questions: {interview.questions}")
print(f"Number of snippets: {len(interview.snippets)}")
```

### Analyzing a Transcript with Custom Questions

```python
from modules.transcriptanalyzer import TranscriptAnalyzer

# Initialize the analyzer
analyzer = TranscriptAnalyzer(max_threads=4)

# Analyze a transcript with custom questions
answers = analyzer.analyze(
    questions_path="samples/questions.json",
    transcript_path="samples/interview.txt"
)

# Print the answers
for answer in answers:
    print(answer)
```

### Example Questions JSON Format

```json
[
  "What are the main challenges mentioned in the interview?",
  "What solutions were proposed?",
  "What is the interviewee's background and experience?"
]
```

## Core Components

### Preprocessor

The `Preprocessor` class handles the initial processing of interview transcripts:

- Splits large transcripts into manageable chunks
- Identifies speakers and their dialogues
- Extracts questions from the interview
- Creates a structured `Interview` object

### TranscriptAnalyzer

The `TranscriptAnalyzer` class performs in-depth analysis of preprocessed interviews:

- Cleans and summarizes interview chunks
- Extracts relevant information based on questions
- Generates comprehensive answers
- Stores processed data in ChromaDB for retrieval

## Docker Integration

The project includes a Docker Compose configuration for Neo4j integration:

```bash
docker-compose up -d
```

This will start a Neo4j instance that can be used for graph-based knowledge representation.

## Development

This project uses pre-commit hooks for code quality:

```bash
pre-commit install
```

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
