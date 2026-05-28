# LiteLLM - Setup Local

## Cerințe
- Docker & Docker Compose
- Python 3.11+ cu `httpx` (`pip install httpx`)

## Pornire Rapidă

1. Pornește serviciile (prima rulare descarcă modelele, poate dura câteva minute):
   ```bash
   docker-compose up -d
   ```

2. Rulează testul:
   ```bash
   python test_litellm.py
   ```

## Modele Disponibile
`qwen`, `gemma`, `llama`, `mistral`, `phi`

## Adăugare Modele Noi

Adaugă modelul în **două locuri**:

1. **docker-compose.yml** - în comanda serviciului `ollama-init`:
   ```yaml
   for model in qwen2.5:3b gemma2:2b llama3.2:3b mistral:7b phi3:3.8b MODEL_NOU; do
   ```

2. **litellm-config.yaml** - adaugă o intrare nouă:
   ```yaml
   - model_name: modelulmeu
     litellm_params:
       model: ollama/model-nou:tag
       api_base: http://ollama:11434
   ```

3. Repornește pentru a descărca și înregistra:
   ```bash
   docker-compose down && docker-compose up -d
   ```

## Adăugare Gemini

Adaugă în **litellm-config.yaml**:
```yaml
- model_name: gemini
  litellm_params:
    model: gemini/gemini-2.5-flash
    api_key: os.environ/GEMINI_API_KEY
```

Adaugă cheia API în `.env`:
```
GEMINI_API_KEY=cheia-ta-aici
```

Folosește în cod:
```python
model="gemini"
```
