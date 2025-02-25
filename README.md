# Transkriptanalyse-Tool

Ein leistungsstarkes Tool zur Analyse von Interview-Transkripten mit Hilfe von KI. Das Tool verarbeitet Transkripte, extrahiert relevante Informationen basierend auf Fragen und erstellt eine umfassende Auswertung.

## Funktionen

- Vorverarbeitung von Transkripten
- Erkennung von Sprecherwechseln
- Zerlegung des Texts in semantisch sinnvolle Chunks
- Relevante Informationsextraktion basierend auf vordefinierten Fragen
- Vektorbasierte Suche mit ChromaDB
- Generierung einer strukturierten Gesamtauswertung

## Anforderungen

Das Projekt benötigt folgende Python-Pakete:

langchain_text_splitters>=0.0.1
pandas>=2.0.0
openai>=1.1.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
chromadb>=0.4.6

## Installation

# Repository klonen
git clone https://github.com/username/transkriptanalyse.git
cd transkriptanalyse

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install langchain_text_splitters pandas openai python-dotenv tiktoken chromadb

## Konfiguration

Erstellen Sie eine `.env`-Datei im Stammverzeichnis mit folgenden Variablen:

```
AZURE_API_KEY=your_azure_openai_api_key
AZURE_API_VERSION=your_azure_openai_api_version
AZURE_ENDPOINT=your_azure_openai_endpoint
AZURE_EMBEDDING_KEY=your_azure_embedding_key
AZURE_EMBEDDING_ENDPOINT=your_azure_embedding_endpoint
```

## Datenstruktur

Das Tool erwartet folgende Dateien:

- `data/sample_transcript.txt`: Das zu analysierende Interview-Transkript
- `data/questions.json`: Die für die Analyse relevanten Fragen im JSON-Format

## Verwendung

```
# Hauptanalyse ausführen
python analyze.py
```

## Ausgabe

Das Tool generiert folgende Ausgabedateien:

- `extracted_information.json`: Extrahierte Informationen für jede Frage
- `final_answer.txt`: Strukturierte Gesamtauswertung des Interviews

## Projektstruktur

```
├── analyze.py               # Hauptskript für die Transkriptanalyse
├── prompts.py               # Vorlagen für GPT-Prompts
├── data/
│   ├── sample_transcript.txt # Interview-Transkript
│   └── questions.json       # Analysefragen
├── chroma_db/               # Speicherort für die Vektordatenbank
├── extracted_information.json # Generierte Zwischenergebnisse
└── final_answer.txt         # Finale Auswertung
```

## Hinweise zur Anpassung

- Passen Sie die Chunk-Größe in `analyze.py` an, falls die Transkripte sehr lang oder kurz sind
- Die Prompts in `prompts.py` können für unterschiedliche Analyseanforderungen angepasst werden
- Bei großen Transkripten beachten Sie das Token-Limit von GPT-4o (125.000 Tokens)

