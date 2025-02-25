speakers_prompt = """
Hier ist ein Transkript eines Interviews. Jeder Sprecher ist mit zwei Großbuchstaben (z. B. AB, CD) gekennzeichnet. Extrahiere alle einzigartigen Speaker und gib sie als kommagetrennte Liste aus, ohne Wiederholungen.

Wichtige Anweisungen:

Gib nur die Liste der einzigartigen Speaker zurück, getrennt durch ein Komma.
Sortiere die Speaker in der Reihenfolge ihres ersten Auftretens.
Ignoriere jeglichen anderen Text oder Formatierungen.
Es gibt nur Speaker mit 2 Großbuchstaben. Es gibt keine Speaker mit 3 oder 1 Großbuchstaben.
Jeder Speaker muss mindestens 2 mal im Transkript auftauchen.
Es handelt sich um ein Interview, nenne nur die 2 wichtigsten Speaker.
Beispiel für die gewünschte Ausgabe:
AB, CD, EF, GH
"""

extraction_system_prompt = """
Hier ist ein Abschnitt aus einem Interview sowie eine dazugehörige Frage. Deine Aufgabe ist es, aus dem Abschnitt jede mögliche relevante Information, jedes Detail und jede potenzielle Interpretation zu extrahieren, die zur Beantwortung der Frage beitragen kann. Gehe in die tiefstmögliche Analyse, lasse nichts aus, fasse nichts zusammen – gib alles, was aus diesem Abschnitt herausgeholt werden kann. Führe auch eine kritische Reflexion durch, ob bestimmte Aussagen mehrdeutig sind, ob es versteckte Implikationen gibt oder ob der Kontext auf alternative Interpretationen hindeutet. Falls es Widersprüche gibt, benenne und analysiere sie. Achte darauf, dass keine relevante Nuance verloren geht.
"""

extraction_prompt = """

Interview-Ausschnitt:
{interview_chunk}

Frage:
{interview_question}

Relevante andere Chunks:
{relevant_chunks}
"""

cleaning_prompt = "Bereinige diesen Text von unnötigen Zeichen und Füllwörtern. Gebe nur den bereinigten Text zurück, keine anderen Informationen."

summary_prompt = """Analysiere den folgenden Interview-Abschnitt sorgfältig. Identifiziere die zentralen Themen und liste sie als Stichpunkte auf. Ergänze zu jedem Punkt eine kurze Begründung, warum dieses Thema relevant ist. Achte darauf, wer etwas sagt und warum es wichtig ist. Markiere alle Konzepte, Fachbegriffe oder Namen, die im Text vorkommen, und erläutere kurz deren Bedeutung oder Kontext."""

full_format_prompt = """
Kontext:
Du erhältst einen unsortierten Text, der aus interpretierten Interview-Frage-Antwort-Paaren besteht. Deine Aufgabe ist es, diese Informationen in ein einheitliches, strukturiertes Format zu bringen.

Aufgabe:
Bringe die Informationen in ein sinnvolles, einheitliches Format. Jede Interviewfrage soll als Überschrift dienen, gefolgt von einem gut lesbaren Fließtext, der die Interpretation der Antwort enthält.

Format der Antwort:
Deine Antwort soll folgendermaßen strukturiert sein:

# Frage: [Interviewfrage]

[Interpretation als zusammenhängender, gut strukturierter Fließtext]

Achte darauf, dass der Fließtext alle wichtigen Informationen aus der ursprünglichen Interpretation enthält und gut lesbar ist. Verwende einen professionellen Schreibstil und sorge für einen logischen Aufbau der Inhalte.

Hier die Frage, die du in der Auswertung behandeln sollst:
{interview_question}

Stelle sicher, dass die Interpretation vollständig ist und alle relevanten Details enthält, wie z.B. genannte Personen, Zahlen, Orte oder konkrete Beispiele.
Gebe nur die Auswertung im korrekten Format zurück.
"""

