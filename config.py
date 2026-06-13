import os
import re
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Whisper model size setup (default is 'base')
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Whisper device setup (default is 'cpu', can be 'cuda' for GPU execution)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

# Whisper compute type (default is 'int8' for cpu, and 'float16' for cuda if not provided)
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE")
if not WHISPER_COMPUTE_TYPE:
    WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

# Setup Presidio NLP engine with a lightweight model 'en_core_web_sm'
NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"}
    ]
}

# Initialize Presidio NLP Engine
provider = NlpEngineProvider(nlp_configuration=NLP_CONFIG)
nlp_engine = provider.create_engine()

def is_luhn_valid(number_str: str) -> bool:
    """
    Validates a credit card number using the Luhn algorithm (Mod 10).
    """
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total_sum = sum(odd_digits)
    for d in even_digits:
        double_d = d * 2
        total_sum += (double_d - 9) if double_d > 9 else double_d
    return (total_sum % 10) == 0

class LuhnCreditCardRecognizer(EntityRecognizer):
    """
    Custom Presidio recognizer that detects credit card numbers
    by scanning for 13-19 digit patterns (possibly separated by spaces/hyphens)
    and validating them using the Luhn algorithm.
    """
    def __init__(self):
        super().__init__(
            supported_entities=["CREDIT_CARD"],
            supported_language="en"
        )
        # Matches 13 to 19 digits possibly separated by spaces or hyphens
        self.pattern = re.compile(r'\b(?:\d[\s-]*){13,19}\b')

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        # If entities filter is provided and CREDIT_CARD is not in it, skip
        if entities and "CREDIT_CARD" not in entities:
            return results

        for match in self.pattern.finditer(text):
            candidate = match.group(0)
            # Filter candidate down to only digits
            cleaned = "".join(c for c in candidate if c.isdigit())
            if 13 <= len(cleaned) <= 19 and is_luhn_valid(cleaned):
                result = RecognizerResult(
                    entity_type="CREDIT_CARD",
                    start=match.start(),
                    end=match.end(),
                    score=0.95
                )
                results.append(result)
        return results

# Initialize Presidio Analyzer Engine
analyzer_engine = AnalyzerEngine(nlp_engine=nlp_engine)
# Add our custom robust Luhn recognizer to the registry
analyzer_engine.registry.add_recognizer(LuhnCreditCardRecognizer())
