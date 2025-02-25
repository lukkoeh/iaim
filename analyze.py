from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import json
import re
from typing import Dict, List
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

import tiktoken
from prompts import summary_prompt, cleaning_prompt, extraction_system_prompt, extraction_prompt, full_format_prompt
import chromadb
import chromadb.utils.embedding_functions as embedding_functions


def preprocess_transcript(transcript: str) -> str:
    """
    Bereitet das Transkript für die Verarbeitung vor, indem Sprecherwechsel
    klarer markiert werden.

    Args:
        transcript: Das Rohtranskript als String

    Returns:
        Ein vorverarbeitetes Transkript mit klaren Trennzeichen
    """
    print("📝 Starte Vorverarbeitung des Transkripts...")
    # Behandle den ersten Sprecher separat
    if re.match(r"^([A-Z]{1,3})$", transcript.split("\n")[0]):
        first_speaker = transcript.split("\n")[0]
        transcript = transcript.replace(
            f"{first_speaker}\n", f"[SPEAKER:{first_speaker}]\n", 1
        )
        print(f"👤 Erster Sprecher erkannt: {first_speaker}")

    # Finde alle anderen Sprecherwechsel (Muster: Zeilenumbruch, Großbuchstaben, Zeilenumbruch)
    processed = re.sub(r"\n([A-Z]{1,3})\n", r"\n[SPEAKER:\1]\n", transcript)
    speaker_count = len(re.findall(r"\[SPEAKER:([A-Z]{1,3})\]", processed))
    print(f"🔢 Insgesamt {speaker_count} Sprecherwechsel erkannt")
    return processed


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starte Transkriptanalyse")
    print("=" * 50)

    load_dotenv(override=True)
    print("🔄 Umgebungsvariablen geladen")

    master_canvas: List[Dict] = []
    # Create a dataframe from the json file
    json_data: Dict = json.load(open("data/questions.json", "r", encoding="utf-8"))
    questions_df: pd.DataFrame = pd.DataFrame(json_data)
    print(f"❓ Fragen geladen: {len(questions_df)} Einträge gefunden")

    # Read the transcript file and clean it from line breaks and unneccessary spaces
    transcript: str = open("data/sample_transcript.txt", "r", encoding="utf-8").read()
    print(f"📄 Transkript geladen: {len(transcript)} Zeichen")

    # Initialize the azure openai client
    openai_client: AzureOpenAI = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    )
    print("🤖 Azure OpenAI Client initialisiert")

    default_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("AZURE_EMBEDDING_KEY"),
        api_base=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
        api_type="azure",
        api_version="2023-05-15",
        model_name="text-embedding-3-large",
    )
    print("🧠 Embedding-Funktion konfiguriert")
    
    # Create chromadb client with the openai embedding function
    db_client = chromadb.PersistentClient(path="chroma_db")
    print("💾 ChromaDB Client erstellt mit Pfad: chroma_db")
    
    # Create a collection in the chroma db
    print("🗑️ Lösche vorhandene Kollektion 'transcript_chunks_jw' falls vorhanden...")
    try:
        db_client.delete_collection(name="transcript_chunks_jw")
        print("✅ Vorhandene Kollektion gelöscht")
    except Exception as e:
        print(f"ℹ️ Keine vorhandene Kollektion gefunden oder Fehler beim Löschen: {e}")
    
    collection = db_client.create_collection(name="transcript_chunks_jw", embedding_function=default_ef)
    print("📦 Neue ChromaDB Kollektion 'transcript_chunks_jw' erstellt")
    
    # Transkript vorverarbeiten
    print("\n" + "-" * 50)
    print("🔄 Starte Transkriptvorverarbeitung...")
    processed_transcript = preprocess_transcript(transcript)
    print("✅ Transkriptvorverarbeitung abgeschlossen")

    # Zeige das vorverarbeitete Transkript zur Überprüfung
    print("\n📝 Vorverarbeitetes Transkript:")
    print(processed_transcript[:200] + "...")  # Nur die ersten 200 Zeichen
    print("-" * 50)

    # Teile das Transkript nach Sprechern auf
    print("\n✂️ Teile Transkript in Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\\[SPEAKER:", "\n", " "],  # Korrigierter Escape-Charakter
        chunk_size=800,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )
    print(f"⚙️ Text-Splitter konfiguriert: Chunk-Größe={800}, Überlappung={0}")

    chunks = text_splitter.create_documents([processed_transcript])
    print(f"📊 Transkript in {len(chunks)} Chunks aufgeteilt")

    print(f"Anzahl der Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content[:50]}...")  # Zeigt die ersten 50 Zeichen jedes Chunks

    # Optional: Bereinige die Chunks, um Sprecher zu extrahieren
    print("\n🧹 Bereinige Chunks und extrahiere Sprecher...")
    clean_chunks = []
    speaker_stats = {}
    
    for i, chunk in enumerate(chunks):
        print(f"🔄 Bereinige Chunk {i+1}/{len(chunks)} und fasse ihn zusammen (1. Zusammenfassung)")
        text = chunk.page_content
        # Extrahiere den Sprecher aus dem Text (falls vorhanden)
        speaker_match = re.search(r"\[SPEAKER:([A-Z]{1,3})\]", text)
        speaker = speaker_match.group(1) if speaker_match else "UNKNOWN"
        
        # Zähle Sprecher für Statistik
        if speaker in speaker_stats:
            speaker_stats[speaker] += 1
        else:
            speaker_stats[speaker] = 1

        # Entferne die Sprechermarkierung aus dem Text
        clean_text = re.sub(r"\[SPEAKER:[A-Z]{1,3}\]", "", text).strip()
        
        # Bereinige den Text mittels OpenAI um unnötige Zeichen und Füllwörter
        clean_text = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cleaning_prompt},
                {"role": "user", "content": clean_text},
            ],
        )
        clean_text = clean_text.choices[0].message.content
        # Erzeuge eine erste Zusammenfassung jedes Chunks und füge es einfach an den Text mit \n\n zusammen
        summary = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": clean_text},
            ],
        )
        summary = summary.choices[0].message.content
        clean_text = f"{clean_text}\n\n{summary}"

        if clean_text:  # Ignoriere leere Chunks
            clean_chunks.append(clean_text)
            if i < 3 or i >= len(chunks) - 3:  # Zeige die ersten und letzten 3 Chunks
                print(f"📄 Chunk {i+1} (Sprecher: {speaker}): {clean_text[:50]}...")

    print(f"\n✅ Bereinigte Chunks: {len(clean_chunks)}")
    print("👥 Sprecher-Statistik:")
    for speaker, count in speaker_stats.items():
        print(f"  👤 {speaker}: {count} Chunks")

    # clean_chunks ist nun eine Liste von Chunks mit Sprecher und Text
    # Throw the chunks into the chroma db
    print("\n" + "-" * 50)
    print(f"💾 Füge {len(clean_chunks)} Chunks zur ChromaDB hinzu...")
    collection.add(
        ids=[str(i) for i in range(len(clean_chunks))],
        documents=clean_chunks,
    )
    print("✅ Chunks erfolgreich zur ChromaDB hinzugefügt")
    print(f"📊 Anzahl der Dokumente in der Kollektion: {collection.count()}")
    print("=" * 50)
    print("🎉 Transkriptanalyse abgeschlossen")
    print("=" * 50)
    
    extracted_information = []
    
    # cycle through the questions and the chunks and do a rag based extraction of information
    for q, question in enumerate(questions_df["question"], 1):
        print(f"❓ Frage: {question} (Frage {q}/{len(questions_df)})")
        # get the chunks that are relevant to the question
        relevant_chunks = collection.query(
            query_texts=[question],
            n_results=5
        )
        
        # Verwende die relevanten Chunks für die Extraktion
        chunk_texts = relevant_chunks["documents"][0]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": extraction_system_prompt},
                {"role": "user", "content": extraction_prompt.format(interview_chunk="\n".join(chunk_texts), interview_question=question, relevant_chunks=chunk_texts)},
            ],
        )
        response = response.choices[0].message.content
        extracted_information.append({"question": question, "interpretation": response})
            
    # save the extracted information to a json file
    with open("extracted_information.json", "w", encoding="utf-8") as f:
        json.dump(extracted_information, f, ensure_ascii=False)
        print("💾 Extrahierte Informationen in 'extracted_information.json' gespeichert")
    
    # Combine the extracted information into a single string, count its tokens with the tokenizer of gpt-4o using tiktoken, to find out if it fits into the context window
    combined_information = "\n\n".join([f"Frage: {item['question']}\n\nInterpretation: {item['interpretation']}" for item in extracted_information])
    tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(combined_information))
    print(f"🔢 Anzahl der Tokens in der kombinierten Information: {tokens}")
    
    if tokens < 125000:
        print("✅ Die kombinierte Information passt in das Kontextfenster von gpt-4o")
        # Verarbeite jede Frage einzeln
        final_answers = []
        
        for q, question in enumerate(questions_df["question"], 1):
            print(f"🔄 Verarbeite Frage {q}/{len(questions_df)}: {question}")
            
            # Filtere die extrahierten Informationen für diese spezifische Frage
            question_info = next((item for item in extracted_information if item["question"] == question), None)
            
            if question_info:
                # Bereite den Inhalt für diese Frage vor
                question_content = f"Frage: {question_info['question']}\n\nInterpretation: {question_info['interpretation']}"
                
                # Generiere die Antwort für diese einzelne Frage
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": full_format_prompt.format(
                            interview_question=question)},
                        {"role": "user", "content": question_content},
                    ],
                    max_tokens=16384,
                    temperature=1,
                )
                
                question_answer = response.choices[0].message.content
                
                # Setze die Generierung fort, falls die Antwort nicht vollständig ist
                messages_history = [
                    {"role": "system", "content": full_format_prompt.format(
                        interview_question=question)},
                    {"role": "user", "content": question_content},
                    {"role": "assistant", "content": question_answer}
                ]
                
                while response.choices[0].finish_reason != "stop":
                    print(f"🔄 Die Antwort für Frage {q} ist noch nicht vollständig, setze fort...")
                    messages_history.append({"role": "user", "content": "Bitte setze deine Analyse fort..."})
                    
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages_history,
                        max_tokens=16384,
                        temperature=1,
                    )
                    
                    continuation = response.choices[0].message.content
                    question_answer += "\n" + continuation
                    messages_history.append({"role": "assistant", "content": continuation})
                
                # Speichere die Antwort für diese Frage
                final_answers.append({
                    "question": question,
                    "answer": question_answer
                })
                
                tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(question_answer))
                print(f"✅ Frage {q} verarbeitet. Tokens: {tokens}")
        
        # Kombiniere alle Antworten zu einem Gesamtdokument
        final_answer = "# Gesamtauswertung des Interviews\n\n"
        final_answer += "# Einleitung\n\nDieses Dokument enthält die Auswertung eines Interviews basierend auf den gestellten Fragen.\n\n"
        final_answer += "# Hauptteil\n\n"
        
        for item in final_answers:
            # Entferne eventuell vorhandene Einleitungen und Schlussfolgerungen aus den Einzelantworten
            answer_content = item["answer"]
            answer_content = re.sub(r"# Einleitung.*?(?=## Frage|# Hauptteil)", "", answer_content, flags=re.DOTALL)
            answer_content = re.sub(r"# Schlussfolgerung.*", "", answer_content, flags=re.DOTALL)
            
            # Füge die bereinigte Antwort zum Gesamtdokument hinzu
            final_answer += answer_content + "\n\n"
        
        final_answer += "# Schlussfolgerung\n\nDie obige Analyse fasst die wichtigsten Erkenntnisse aus dem Interview zusammen."
        
        # Speichere die Gesamtantwort
        with open("final_answer.txt", "w", encoding="utf-8") as f:
            f.write(final_answer)
            
        tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(final_answer))
        print(f"🔢 Anzahl der Tokens in der finalen Gesamtantwort: {tokens}")
        print("📄 Finale Gesamtantwort in 'final_answer.txt' gespeichert")
    else:
        print("⚠️ Die kombinierte Information passt nicht in den Kontextfenster von gpt-4o")
        # Implementiere eine Alternative für große Inhalte
        print("🔄 Verarbeite Antworten in Batches...")
        # Hier könnte Code für eine Batch-Verarbeitung folgen

# Führe die Informationen zusammen, indem die erste Zusammenfassung immer mit der nächsten verknüpft und erweitert wird.
