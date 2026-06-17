import asyncio
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# Import local modules
from audio_utils import merge_intervals, bleep_audio
from main import app, cleanup_files
from pipeline import run_redaction_pipeline

class TestAudioUtils(unittest.TestCase):
    def test_merge_intervals_empty(self):
        self.assertEqual(merge_intervals([]), [])

    def test_merge_intervals_no_overlap(self):
        intervals = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        expected = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        self.assertEqual(merge_intervals(intervals), expected)

    def test_merge_intervals_overlapping(self):
        intervals = [(1.0, 3.0), (2.0, 4.0), (5.0, 6.0)]
        expected = [(1.0, 4.0), (5.0, 6.0)]
        self.assertEqual(merge_intervals(intervals), expected)

    def test_merge_intervals_adjacent(self):
        intervals = [(1.0, 2.0), (2.0, 3.0)]
        expected = [(1.0, 3.0)]
        self.assertEqual(merge_intervals(intervals), expected)

    def test_merge_intervals_nested(self):
        intervals = [(1.0, 10.0), (2.0, 5.0), (6.0, 8.0)]
        expected = [(1.0, 10.0)]
        self.assertEqual(merge_intervals(intervals), expected)

    def test_merge_intervals_unsorted(self):
        intervals = [(5.0, 6.0), (1.0, 3.0), (2.0, 4.0)]
        expected = [(1.0, 4.0), (5.0, 6.0)]
        self.assertEqual(merge_intervals(intervals), expected)


class TestFFmpegBleep(unittest.IsolatedAsyncioTestCase):
    async def test_bleep_audio_execution(self):
        """
        Creates a short dummy silent audio and runs bleep_audio using real ffmpeg.
        Verifies that the command runs and the output file is generated correctly.
        """
        temp_dir = tempfile.gettempdir()
        dummy_input = os.path.join(temp_dir, f"test_dummy_in.wav")
        dummy_output = os.path.join(temp_dir, f"test_dummy_out.wav")

        # Cleanup pre-existing files
        cleanup_files(dummy_input, dummy_output)

        try:
            # 1. Generate a 3-second dummy silent wav file
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "3", "-y", dummy_input,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            self.assertTrue(os.path.exists(dummy_input), "Dummy input file should be created.")

            # 2. Run bleep_audio on it to bleep the segment [1.0, 2.0]
            await bleep_audio(dummy_input, dummy_output, [(1.0, 2.0)])

            # 3. Assert output exists and is non-empty
            self.assertTrue(os.path.exists(dummy_output), "Bleeped output file should be created.")
            self.assertGreater(os.path.getsize(dummy_output), 0, "Bleeped output file should be non-empty.")

        finally:
            # Cleanup
            cleanup_files(dummy_input, dummy_output)

    async def test_stereo_processing(self):
        """
        Creates a dummy stereo WAV file and verifies the stereo channel extraction
        and bleeping logic.
        """
        from audio_utils import get_audio_channels, extract_mono_channel
        temp_dir = tempfile.gettempdir()
        dummy_stereo = os.path.join(temp_dir, "test_dummy_stereo.wav")
        dummy_left = os.path.join(temp_dir, "test_dummy_left.wav")
        dummy_bleeped = os.path.join(temp_dir, "test_dummy_bleeped_stereo.wav")

        # Cleanup
        cleanup_files(dummy_stereo, dummy_left, dummy_bleeped)

        try:
            # 1. Generate a 3-second dummy stereo file (different frequencies in FL and FR)
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100",
                "-filter_complex", "[0:a][1:a]amerge=inputs=2", "-t", "3", "-y", dummy_stereo,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            self.assertTrue(os.path.exists(dummy_stereo), "Dummy stereo file should be created.")

            # 2. Check channel detection
            channels = await get_audio_channels(dummy_stereo)
            self.assertEqual(channels, 2)

            # 3. Extract Front Left (FL) channel
            await extract_mono_channel(dummy_stereo, dummy_left, "left")
            self.assertTrue(os.path.exists(dummy_left), "Extracted left channel file should be created.")
            left_channels = await get_audio_channels(dummy_left)
            self.assertEqual(left_channels, 1)

            # 4. Bleep the stereo file
            await bleep_audio(dummy_stereo, dummy_bleeped, [(1.0, 2.0)])
            self.assertTrue(os.path.exists(dummy_bleeped), "Bleeped stereo file should be created.")
            bleeped_channels = await get_audio_channels(dummy_bleeped)
            self.assertEqual(bleeped_channels, 2)

        finally:
            cleanup_files(dummy_stereo, dummy_left, dummy_bleeped)


class TestPipelineMapping(unittest.IsolatedAsyncioTestCase):
    @patch("pipeline.get_whisper_model")
    @patch("pipeline.analyzer_engine")
    async def test_run_redaction_pipeline_with_word_timestamps(self, mock_analyzer, mock_get_whisper):
        # Setup mock Whisper segments and words
        mock_word_1 = MagicMock()
        mock_word_1.word = " Hello,"
        mock_word_1.start = 0.0
        mock_word_1.end = 0.5

        mock_word_2 = MagicMock()
        mock_word_2.word = " my"
        mock_word_2.start = 0.5
        mock_word_2.end = 0.8

        mock_word_3 = MagicMock()
        mock_word_3.word = " phone"
        mock_word_3.start = 0.8
        mock_word_3.end = 1.2

        mock_word_4 = MagicMock()
        mock_word_4.word = " is"
        mock_word_4.start = 1.2
        mock_word_4.end = 1.4

        mock_word_5 = MagicMock()
        mock_word_5.word = " 555-0199."
        mock_word_5.start = 1.4
        mock_word_5.end = 2.4

        mock_segment = MagicMock()
        mock_segment.words = [mock_word_1, mock_word_2, mock_word_3, mock_word_4, mock_word_5]
        mock_segment.text = " Hello, my phone is 555-0199."

        mock_whisper = MagicMock()
        # Mocking the transcribe call
        mock_whisper.transcribe.return_value = ([mock_segment], None)
        mock_get_whisper.return_value = mock_whisper

        # Setup mock Presidio Analyzer response
        # " Hello, my phone is 555-0199."
        # Character indices of "555-0199." in " Hello, my phone is 555-0199."
        # " Hello," -> len 7
        # " my" -> len 3 -> total 10
        # " phone" -> len 6 -> total 16
        # " is" -> len 3 -> total 19
        # " 555-0199." -> len 10 -> total 29
        # The character start index of " 555-0199." is 19, end index is 29.
        mock_pii_result = MagicMock()
        mock_pii_result.entity_type = "PHONE_NUMBER"
        mock_pii_result.start = 20  # Inside "555-0199."
        mock_pii_result.end = 28
        mock_pii_result.score = 0.95

        mock_analyzer.analyze.return_value = [mock_pii_result]

        # Run pipeline with full_redact=True to allow non-credit-card PII (PHONE_NUMBER)
        bleep_segments = await run_redaction_pipeline("dummy_audio.wav", full_redact=True)

        # We expect only the word " 555-0199." to be bleeped, which has timestamps [1.4, 2.4]
        self.assertEqual(bleep_segments, [(1.4, 2.4)])
        mock_analyzer.analyze.assert_called_once_with(
            text=" Hello, my phone is 555-0199.",
            language="en",
            entities=None
        )

    @patch("pipeline.get_whisper_model")
    @patch("pipeline.analyzer_engine")
    async def test_run_redaction_pipeline_default_only_credit_card(self, mock_analyzer, mock_get_whisper):
        # Setup mock Whisper segment and words
        mock_word = MagicMock()
        mock_word.word = " 4444333322221111"
        mock_word.start = 0.0
        mock_word.end = 1.0

        mock_segment = MagicMock()
        mock_segment.words = [mock_word]
        mock_segment.text = " 4444333322221111"

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([mock_segment], None)
        mock_get_whisper.return_value = mock_whisper

        mock_analyzer.analyze.return_value = []

        # Run pipeline with default options (full_redact=False)
        await run_redaction_pipeline("dummy_audio.wav")

        # Verify that Presidio was called requesting ONLY "CREDIT_CARD" entities
        mock_analyzer.analyze.assert_called_once_with(
            text=" 4444333322221111",
            language="en",
            entities=["CREDIT_CARD"]
        )

    @patch("pipeline.get_whisper_model")
    @patch("pipeline.analyzer_engine")
    async def test_run_redaction_pipeline_continuous_bleeping(self, mock_analyzer, mock_get_whisper):
        # Setup mock Whisper segment and multiple words
        mock_word_1 = MagicMock()
        mock_word_1.word = " 1,"
        mock_word_1.start = 1.0
        mock_word_1.end = 1.5

        mock_word_2 = MagicMock()
        mock_word_2.word = " 2,"
        mock_word_2.start = 1.6
        mock_word_2.end = 2.1

        mock_word_3 = MagicMock()
        mock_word_3.word = " 3."
        mock_word_3.start = 2.2
        mock_word_3.end = 2.7

        mock_segment = MagicMock()
        mock_segment.words = [mock_word_1, mock_word_2, mock_word_3]
        mock_segment.text = " 1, 2, 3."

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([mock_segment], None)
        mock_get_whisper.return_value = mock_whisper

        # Setup mock Presidio Analyzer response
        mock_pii_result = MagicMock()
        mock_pii_result.entity_type = "CREDIT_CARD"
        mock_pii_result.start = 1  # Matches "1, 2, 3."
        mock_pii_result.end = 9
        mock_pii_result.score = 0.95

        mock_analyzer.analyze.return_value = [mock_pii_result]

        # Run pipeline
        bleep_segments = await run_redaction_pipeline("dummy_audio.wav")

        # We expect a single continuous interval covering all matched words
        self.assertEqual(bleep_segments, [(1.0, 2.7)])


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "service": "Redactify"})

    @patch("main.run_redaction_pipeline", new_callable=AsyncMock)
    @patch("main.bleep_audio", new_callable=AsyncMock)
    def test_redact_audio_endpoint_success(self, mock_bleep_audio, mock_pipeline):
        # Configure Mocks
        mock_pipeline.return_value = [(1.0, 2.0)]
        
        # We mock bleep_audio to create a dummy output file on disk so FileResponse doesn't fail
        async def mock_bleep_side_effect(input_path, output_path, segments):
            with open(output_path, "wb") as f:
                f.write(b"fake_bleeped_audio")
        mock_bleep_audio.side_effect = mock_bleep_side_effect

        # Create a dummy upload file
        temp_dir = tempfile.gettempdir()
        dummy_upload_file = os.path.join(temp_dir, "to_upload.wav")
        with open(dummy_upload_file, "wb") as f:
            f.write(b"fake_input_audio")

        try:
            with open(dummy_upload_file, "rb") as f:
                response = self.client.post(
                    "/redact-audio",
                    files={"file": ("to_upload.wav", f, "audio/wav")}
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"fake_bleeped_audio")
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertTrue(response.headers["content-disposition"].startswith("attachment; filename="))

        finally:
            cleanup_files(dummy_upload_file)

    def test_redact_text_endpoint_success_credit_card(self):
        # Default: redact only CREDIT_CARD
        text_content = b"Agent: Hello, how can I help you?\nCustomer: Yes, my card number is 4444 3333 2222 1111."
        response = self.client.post(
            "/redact-text",
            files={"file": ("transcript.txt", text_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        self.assertTrue(response.headers["content-disposition"].startswith("attachment; filename="))
        
        redacted_body = response.content.decode("utf-8")
        self.assertIn("Customer: Yes, my card number is <CREDIT_CARD>.", redacted_body)
        self.assertNotIn("4444 3333 2222 1111", redacted_body)

    def test_redact_text_endpoint_success_full_redact(self):
        # full_redact=True: redacts other PII like phone numbers and names
        text_content = b"Agent: Hi John, is 555-555-0199 your number?\nCustomer: Yes it is."
        response = self.client.post(
            "/redact-text?full_redact=true",
            files={"file": ("transcript.txt", text_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        
        redacted_body = response.content.decode("utf-8")
        # Ensure name and phone are redacted
        self.assertNotIn("John", redacted_body)
        self.assertNotIn("555-555-0199", redacted_body)
        # Verify placeholders exist (either <PERSON> or <PHONE_NUMBER>)
        self.assertTrue("<PERSON>" in redacted_body)
        self.assertTrue("<PHONE_NUMBER>" in redacted_body)

    def test_redact_text_endpoint_missing_filename(self):
        response = self.client.post(
            "/redact-text",
            files={"file": ("", b"some text content", "text/plain")}
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Expected UploadFile", response.json()["detail"][0]["msg"])


class TestLuhnRecognizer(unittest.TestCase):
    def test_luhn_validation(self):
        from config import is_luhn_valid
        # Valid test card number passing Luhn
        self.assertTrue(is_luhn_valid("4444333322221111"))
        # Invalid number (fails Luhn)
        self.assertFalse(is_luhn_valid("4444333322221112"))

    def test_luhn_recognizer_detection(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()
        
        # Test detection with spaces and hyphens
        text = "My card number is 4444 3333 2222 1111 and my backup is 4444-3333-2222-1111."
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        self.assertEqual(len(results), 2)
        
        self.assertEqual(results[0].entity_type, "CREDIT_CARD")
        self.assertEqual(results[1].entity_type, "CREDIT_CARD")

    def test_luhn_recognizer_detection_commas_and_periods(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()
        
        # Test detection with spaces, commas, and periods
        text = "My card number is 4, 4, 4, 4. 3, 3, 3, 3. 2, 2, 2, 2. 1, 1, 1, 1."
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_type, "CREDIT_CARD")

    def test_luhn_recognizer_detection_word_separated(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()

        # The user's exact transcription example
        text = (
            "All right, I wanted to give my credit card number. All right, go ahead. "
            "All right, let's see here. One, two, three, four. Okay. Four, three, two, one. "
            "Mm-hmm. One, two, three, four. Go ahead. Four, three, two, one. All right, perfect."
        )
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_type, "CREDIT_CARD")
        # Ensure the matched range covers from the first digit word "One" to the last "one"
        matched_text = text[results[0].start:results[0].end]
        self.assertTrue(matched_text.startswith("One"))
        self.assertTrue(matched_text.endswith("one"))

    def test_luhn_recognizer_detection_multipliers(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()

        # Test "double" and "triple" word multipliers in a card number
        text = "card is double four double four double three double three double two double two double one double one."
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        self.assertEqual(len(results), 1)
        matched_text = text[results[0].start:results[0].end]
        self.assertTrue(matched_text.startswith("double"))
        self.assertTrue(matched_text.endswith("one"))

    def test_luhn_recognizer_detection_tens_and_teens(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()

        # Test "tens" and "teens" words
        text = "card is forty-four forty-four thirty-three thirty-three twenty-two twenty-two eleven eleven"
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        self.assertEqual(len(results), 1)
        matched_text = text[results[0].start:results[0].end]
        self.assertTrue(matched_text.startswith("forty"))
        self.assertTrue(matched_text.endswith("eleven"))

    def test_luhn_recognizer_interrupted_with_words(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()

        # The user's exact failing scenario with conversational interruption and "four" count word
        text = (
            "My visa card? Yeah, it's four transfers of four. So, 1234, then 5678. "
            "Hold on. Let me look at it. Okay, 912. And the last four digits are 3456."
        )
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_type, "CREDIT_CARD")
        matched_text = text[results[0].start:results[0].end]
        self.assertTrue(matched_text.startswith("four"))
        self.assertTrue(matched_text.endswith("3456"))

    def test_conversational_variations(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()
        
        scenarios = [
            # Scenario A: Spelling digits one-by-one with pauses & filler word "oh" (0)
            {
                "text": "My card is four, oh, two, zero, zero, one, one, two, two, two, zero, three, three, zero, eight.",
                "expected_count": 1,
                "desc": "Single digit spelling with 'oh' representing '0'"
            },
            # Scenario B: Large conversational interruption with multipliers
            {
                "text": "The Visa is four four four four. Yeah, hold on, someone is at the door. Okay, back. Then it's three three three three, double two double two, and double one double one.",
                "expected_count": 1,
                "desc": "Conversational interruption with multipliers"
            },
            # Scenario C: Phonetic homophones (to -> 2, for -> 4, ate -> 8)
            {
                "text": "Card is for, for, for, for, three, three, three, three, to, to, to, to, one, one, one, one.",
                "expected_count": 1,
                "desc": "Phonetic homophones transcribed by Whisper"
            },
            # Scenario D: Natural grouped readings (tens + single digits)
            {
                "text": "The account is thirty-seven eighty-two, zero-eleven, twenty-two, twenty, thirty-three, zero-eight.",
                "expected_count": 1,
                "desc": "Grouped tens and teens Amex card reading"
            }
        ]
        
        for s in scenarios:
            with self.subTest(desc=s["desc"]):
                results = recognizer.analyze(s["text"], entities=["CREDIT_CARD"])
                self.assertEqual(len(results), s["expected_count"])

    def test_luhn_recognizer_price_discussions_prevented(self):
        from config import LuhnCreditCardRecognizer
        recognizer = LuhnCreditCardRecognizer()

        # A sequence of prices representing "7000000000000005" which is Luhn-valid but starts with '7'
        text = (
            "Okay, the first quote is seventy dollars, then zero, zero, zero, zero, "
            "zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, and the last is five."
        )
        results = recognizer.analyze(text, entities=["CREDIT_CARD"])
        # Should not be redacted because it starts with '7' (which is not a valid credit card brand prefix)
        self.assertEqual(len(results), 0)


class TestWhisperConfiguration(unittest.TestCase):
    @patch("pipeline.WhisperModel")
    def test_get_whisper_model_configuration(self, mock_whisper_model_class):
        import pipeline
        # Save old cache and force reset
        old_model = pipeline._whisper_model
        pipeline._whisper_model = None
        
        try:
            # Load model
            pipeline.get_whisper_model()
            
            # Assert that WhisperModel was instantiated with configured values
            mock_whisper_model_class.assert_called_once_with(
                pipeline.WHISPER_MODEL_SIZE,
                device=pipeline.WHISPER_DEVICE,
                compute_type=pipeline.WHISPER_COMPUTE_TYPE
            )
        finally:
            # Restore cache
            pipeline._whisper_model = old_model


class TestAppLifespan(unittest.TestCase):
    @patch("main.get_whisper_model")
    def test_lifespan_eagerly_loads_whisper_model(self, mock_get_whisper_model):
        """
        Verify that using TestClient as a context manager triggers the lifespan startup
        which eagerly loads the Whisper model.
        """
        with TestClient(app) as client:
            pass
        # Assert that get_whisper_model was called during the lifespan startup
        mock_get_whisper_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
