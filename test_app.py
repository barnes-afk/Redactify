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

        # Run pipeline
        bleep_segments = await run_redaction_pipeline("dummy_audio.wav")

        # We expect only the word " 555-0199." to be bleeped, which has timestamps [1.4, 2.4]
        self.assertEqual(bleep_segments, [(1.4, 2.4)])


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


if __name__ == "__main__":
    unittest.main()
