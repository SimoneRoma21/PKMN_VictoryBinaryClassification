#!/bin/bash

ENV_DIR="venv"

# Crea l'ambiente virtuale se non esiste
if [ ! -d "$ENV_DIR" ]; then
    echo "Ambiente virtuale non trovato. Creo '$ENV_DIR'..."
    python3 -m venv "$ENV_DIR"
else
    echo "Ambiente virtuale '$ENV_DIR' già presente."
fi

# Attiva l'ambiente virtuale
echo "Attivo l'ambiente virtuale..."
source "$ENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Ambiente virtuale attivato: $VIRTUAL_ENV"
else
    echo "Errore: ambiente virtuale NON attivato!"
fi

# Installa le dipendenze da requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installo le dipendenze da requirements.txt..."
    pip install -r requirements.txt
else
    echo "File requirements.txt non trovato. Nessuna dipendenza installata."
fi