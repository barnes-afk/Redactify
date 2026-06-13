import os
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Whisper model size setup (default is 'base')
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Setup Presidio NLP engine with a lightweight model 'en_core_web_sm'
# to minimize resource consumption and speed up downloads.
NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"}
    ]
}

# Initialize Presidio Analyzer Engine with custom NLP engine
provider = NlpEngineProvider(nlp_configuration=NLP_CONFIG)
nlp_engine = provider.create_engine()
analyzer_engine = AnalyzerEngine(nlp_engine=nlp_engine)
