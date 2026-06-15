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

SINGLE_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "nought": "0", "nil": "0", "none": "0",
    "one": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3",
    "four": "4", "for": "4", "fore": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9"
}

TEENS_WORDS = {
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19"
}

TENS_WORDS = {
    "twenty": "2",
    "thirty": "3",
    "forty": "4",
    "fifty": "5",
    "sixty": "6",
    "seventy": "7",
    "eighty": "8",
    "ninety": "9"
}

MULTIPLIERS = {
    "double": 2,
    "triple": 3
}

class LuhnCreditCardRecognizer(EntityRecognizer):
    """
    Custom Presidio recognizer that detects credit card numbers
    by scanning for 13-19 digit patterns (possibly represented as word numbers or digits,
    and possibly separated by spaces, hyphens, commas, periods, or other words)
    and validating them using the Luhn algorithm.
    """
    def __init__(self):
        super().__init__(
            supported_entities=["CREDIT_CARD"],
            supported_language="en"
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        # If entities filter is provided and CREDIT_CARD is not in it, skip
        if entities and "CREDIT_CARD" not in entities:
            return results

        # 1. Tokenize text into alphanumeric word tokens
        tokens = []
        for m in re.finditer(r'\b\w+\b', text):
            word = m.group(0)
            tokens.append({
                "word": word,
                "lower": word.lower(),
                "start": m.start(),
                "end": m.end()
            })

        # 2. Extract digit tokens and map them to their digit string representations
        digit_items = []
        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            word_lower = token["lower"]
            
            # Raw digit string
            if word_lower.isdigit():
                digit_items.append({
                    "digit_str": word_lower,
                    "start": token["start"],
                    "end": token["end"],
                    "token_idx": i
                })
                i += 1
                continue
                
            # Multiplier (double/triple)
            if word_lower in MULTIPLIERS:
                mult = MULTIPLIERS[word_lower]
                if i + 1 < n:
                    next_token = tokens[i + 1]
                    next_word = next_token["lower"]
                    next_digit = None
                    if next_word.isdigit():
                        next_digit = next_word[0]
                    elif next_word in SINGLE_DIGIT_WORDS:
                        next_digit = SINGLE_DIGIT_WORDS[next_word]
                        
                    if next_digit is not None:
                        digit_items.append({
                            "digit_str": next_digit * mult,
                            "start": token["start"],
                            "end": next_token["end"],
                            "token_idx": i + 1
                        })
                        i += 2
                        continue
                        
            # Single digit words
            if word_lower in SINGLE_DIGIT_WORDS:
                digit_items.append({
                    "digit_str": SINGLE_DIGIT_WORDS[word_lower],
                    "start": token["start"],
                    "end": token["end"],
                    "token_idx": i
                })
                i += 1
                continue
                
            # Teens words
            if word_lower in TEENS_WORDS:
                digit_items.append({
                    "digit_str": TEENS_WORDS[word_lower],
                    "start": token["start"],
                    "end": token["end"],
                    "token_idx": i
                })
                i += 1
                continue
                
            # Tens words
            if word_lower in TENS_WORDS:
                tens_digit = TENS_WORDS[word_lower]
                if i + 1 < n:
                    next_token = tokens[i + 1]
                    next_word = next_token["lower"]
                    if next_word in SINGLE_DIGIT_WORDS:
                        digit_str = tens_digit + SINGLE_DIGIT_WORDS[next_word]
                        digit_items.append({
                            "digit_str": digit_str,
                            "start": token["start"],
                            "end": next_token["end"],
                            "token_idx": i + 1
                        })
                        i += 2
                        continue
                # Otherwise, it's just the tens word by itself (e.g., "forty" -> "40")
                digit_items.append({
                    "digit_str": tens_digit + "0",
                    "start": token["start"],
                    "end": token["end"],
                    "token_idx": i
                })
                i += 1
                continue
                
            i += 1

        # 3. Group digit items that are "close" to each other
        MAX_GAP_TOKENS = 3
        candidate_groups = []
        current_group = []
        
        for item in digit_items:
            if not current_group:
                current_group.append(item)
            else:
                last_item = current_group[-1]
                gap = item["token_idx"] - last_item["token_idx"] - 1
                if gap <= MAX_GAP_TOKENS:
                    current_group.append(item)
                else:
                    candidate_groups.append(current_group)
                    current_group = [item]
        if current_group:
            candidate_groups.append(current_group)

        # 4. Find all Luhn-valid sub-segments within each group
        detected_ranges = []
        for group in candidate_groups:
            # We look at all possible sub-segments of this group
            for start_idx in range(len(group)):
                for end_idx in range(start_idx + 1, len(group) + 1):
                    sub_items = group[start_idx:end_idx]
                    combined = "".join(item["digit_str"] for item in sub_items)
                    if 13 <= len(combined) <= 19:
                        if is_luhn_valid(combined):
                            detected_ranges.append({
                                "start": sub_items[0]["start"],
                                "end": sub_items[-1]["end"]
                            })

        # 5. Merge any overlapping or adjacent detected character ranges
        if not detected_ranges:
            return results

        sorted_ranges = sorted(detected_ranges, key=lambda r: r["start"])
        merged_ranges = []
        curr = sorted_ranges[0]
        for nxt in sorted_ranges[1:]:
            if nxt["start"] <= curr["end"]:
                curr["end"] = max(curr["end"], nxt["end"])
            else:
                merged_ranges.append(curr)
                curr = nxt
        merged_ranges.append(curr)

        # 6. Convert merged ranges to Presidio RecognizerResult objects
        for r in merged_ranges:
            results.append(
                RecognizerResult(
                    entity_type="CREDIT_CARD",
                    start=r["start"],
                    end=r["end"],
                    score=0.95
                )
            )
        return results

# Initialize Presidio Analyzer Engine
analyzer_engine = AnalyzerEngine(nlp_engine=nlp_engine)
# Add our custom robust Luhn recognizer to the registry
analyzer_engine.registry.add_recognizer(LuhnCreditCardRecognizer())
