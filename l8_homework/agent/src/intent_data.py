"""
L8 Task 3 - Date de antrenare pentru Intent Classifier
=======================================================

Clasificăm intenția unui query în 3 clase:
  • search    → utilizatorul caută / vrea să găsească informații
  • extract   → utilizatorul vrea să extragă câmpuri / valori structurate
  • summarize → utilizatorul vrea un rezumat / sinteză

Datele sunt în limba română (domeniul: analiză documente / achiziții).
Set împărțit în train/test mai jos.
"""

# (query, label)
TRAIN_DATA: list[tuple[str, str]] = [
    # --- search ---
    ("Caută toate facturile de la TechSoft", "search"),
    ("Găsește contractele semnate în 2024", "search"),
    ("Unde apare furnizorul DataPro în documente?", "search"),
    ("Arată-mi documentele despre achiziții directe", "search"),
    ("Ce documente menționează CloudNet?", "search"),
    ("Caut informații despre raportul Q1", "search"),
    ("Vreau să găsesc anunțurile de inițiere licitație", "search"),
    ("Identifică toate facturile din luna aprilie", "search"),
    ("Există vreun contract cu SecureIT?", "search"),
    ("Localizează clauzele de penalizare din contracte", "search"),
    ("Care documente conțin valori în EUR?", "search"),
    ("Arată facturile cu valoare peste 10000 RON", "search"),
    ("Caută numele clienților din baza de date", "search"),
    ("Găsește contractul numărul 002", "search"),
    ("Ce fișiere sunt despre WebDev?", "search"),

    # --- extract ---
    ("Extrage suma totală din factura TechSoft", "extract"),
    ("Extrage numele și CUI-ul furnizorului", "extract"),
    ("Care e data și numărul contractului?", "extract"),
    ("Scoate valoarea în RON din document", "extract"),
    ("Extrage adresa de email a clientului DataPro", "extract"),
    ("Vreau câmpurile: furnizor, sumă, dată", "extract"),
    ("Ce contact are DataPro?", "extract"),
    ("Extrage codul CPV din anunț", "extract"),
    ("Scoate-mi numărul de telefon al clientului", "extract"),
    ("Care e valoarea estimată a licitației?", "extract"),
    ("Extrage toate datele de facturare", "extract"),
    ("Dă-mi CUI-ul autorității contractante", "extract"),
    ("Care e totalul facturilor TechSoft?", "extract"),
    ("Extrage tipul procedurii din anunț", "extract"),
    ("Scoate moneda și valoarea din contract", "extract"),

    # --- summarize ---
    ("Rezumă raportul Q1 2024", "summarize"),
    ("Fă-mi un sumar al contractului cu CloudNet", "summarize"),
    ("Pe scurt, despre ce e raportul Q4?", "summarize"),
    ("Sintetizează principalele clauze ale contractului", "summarize"),
    ("Dă-mi un rezumat al activității cu DataPro", "summarize"),
    ("Rezumă în câteva fraze documentul", "summarize"),
    ("Care sunt punctele cheie ale raportului?", "summarize"),
    ("Fă o sinteză a facturilor din 2024", "summarize"),
    ("Rezumă pe scurt relația cu furnizorul SecureIT", "summarize"),
    ("Vreau o privire de ansamblu asupra contractelor", "summarize"),
    ("Sumarizează concluziile raportului trimestrial", "summarize"),
    ("Spune-mi pe scurt ce conține acest document", "summarize"),
    ("Rezumatul executiv al raportului Q1", "summarize"),
    ("Sintetizează termenii principali ai contractului", "summarize"),
    ("Dă-mi esența documentului în 3 fraze", "summarize"),
]

# Set de test separat (nu apare în train) pentru evaluarea corectă a accuracy-ului.
TEST_DATA: list[tuple[str, str]] = [
    # search
    ("Caută facturile de la SecureIT", "search"),
    ("Găsește toate documentele din martie", "search"),
    ("Unde sunt menționate penalitățile?", "search"),
    ("Arată-mi anunțurile de licitație recente", "search"),
    ("Există documente despre raportul anual?", "search"),
    # extract
    ("Extrage suma totală a contractului CloudNet", "extract"),
    ("Care e emailul de contact al clientului?", "extract"),
    ("Scoate data semnării contractului", "extract"),
    ("Extrage codul CPV principal", "extract"),
    ("Dă-mi valoarea în EUR din factură", "extract"),
    # summarize
    ("Rezumă contractul cu TechSoft", "summarize"),
    ("Fă-mi un sumar al raportului Q4 2023", "summarize"),
    ("Care sunt punctele principale ale documentului?", "summarize"),
    ("Sintetizează relația comercială cu CloudNet", "summarize"),
    ("Dă-mi o privire de ansamblu pe scurt", "summarize"),
]

LABELS = ["search", "extract", "summarize"]
