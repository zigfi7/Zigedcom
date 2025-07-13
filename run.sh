#!/bin/bash
#
# Skrypt do uruchamiania narzędzia zigedcom w wirtualnym środowisku Python.
# Automatycznie tworzy środowisko i instaluje zależności, jeśli jest to potrzebne.
#

# Nazwa katalogu dla wirtualnego środowiska
VENV_DIR="venv"

# Sprawdzenie, czy python3 jest dostępny
if ! command -v python3 &> /dev/null
then
    echo "Błąd: python3 nie jest zainstalowany lub nie ma go w ścieżce PATH."
    exit 1
fi

# Tworzenie wirtualnego środowiska, jeśli nie istnieje
if [ ! -d "$VENV_DIR" ]; then
    echo "Tworzenie wirtualnego środowiska w katalogu '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Błąd: Nie udało się utworzyć wirtualnego środowiska."
        exit 1
    fi
fi

# Aktywacja wirtualnego środowiska
# Ten skrypt jest dla systemów Linux/macOS. Dla Windows składnia to: .\\$VENV_DIR\\Scripts\\activate
echo "Aktywacja wirtualnego środowiska..."
source "$VENV_DIR/bin/activate"

# Instalacja wymaganych bibliotek
echo "Instalowanie/weryfikowanie zależności..."
pip install sqlalchemy psycopg2-binary fuzzywuzzy "python-levenshtein>=0.12" --quiet
if [ $? -ne 0 ]; then
    echo "Błąd: Nie udało się zainstalować zależności."
    deactivate
    exit 1
fi

# Uruchomienie głównego skryptu Pythona z przekazanymi argumentami
echo "Uruchamianie zigedcom..."
echo "----------------------------------------"
python zigedcom.py "$@"
echo "----------------------------------------"

# Dezaktywacja środowiska po zakończeniu pracy skryptu
deactivate
echo "Praca zakończona. Środowisko wirtualne zostało zdezaktywowane."
