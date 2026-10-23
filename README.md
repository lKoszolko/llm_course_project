Projekt z kursu Modele językowe (LLM Basics)

Repozytorium zawiera kod i środowisko rozwijane w ramach kursu LLM.

Instrukcja odtworzenia środowiska

Aby uruchomić projekt lokalnie, postępuj zgodnie z poniższymi krokami. Wymagany jest Python w wersji >= 3.11.

Utworzenie środowiska wirtualnego
W głównym katalogu projektu uruchom terminal i wpisz:

python -m venv .venv


Aktywacja środowiska

Na systemie Windows (PowerShell):

.\.venv\Scripts\Activate.ps1


Na systemach Linux / macOS:

source .venv/bin/activate


Instalacja zależności
Zainstaluj wymagane pakiety z pliku blokady (rekomendowane):

python -m pip install -r requirements-lock.txt


(Alternatywnie, jeśli dodajesz nowe pakiety, użyj requirements.txt)

Instalacja pakietu lokalnego
Aby moduł llm_course był poprawnie wykrywany w całym projekcie (np. przez testy), zainstaluj go w trybie edytowalnym:

python -m pip install -e .


Konfiguracja zmiennych środowiskowych

Skopiuj plik szablonu sekretów:

cp .env.template .env


Otwórz nowo utworzony plik .env i uzupełnij klucze API (np. OPENAI_API_KEY, GEMINI_API_KEY).

Tryb działania aplikacji (lokalny model lub API) możesz zmieniać zmienną LLM_MODE.

Uruchamianie testów

Środowisko jest skonfigurowane do korzystania z pytest. Aby uruchomić testy, po prostu wpisz w konsoli:

pytest
