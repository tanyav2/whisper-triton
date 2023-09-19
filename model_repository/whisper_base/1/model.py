from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
)
import triton_python_backend_utils as pb_utils
import numpy as np
import io
import torchaudio


class TritonPythonModel:
    def initialize(self, args):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-base",
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base",
            device_map="auto"
        )
        self.model.config.forced_decoder_ids = None
        self.tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-base", language="English", task="transcribe"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            audio_bytes = pb_utils.get_input_tensor_by_name(request, "audio").as_numpy()[0]

            # Preprocessing
            audio, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                audio = resampler(audio)
                audio = audio.squeeze(0)

            # Inference
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).to("cuda")
            predicted_ids = self.model.generate(inputs.input_features)
            transcription = self.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            # Postprocessing
            transcription_str = transcription[0]
            output_tensor = pb_utils.Tensor(
                "transcription", np.array([transcription_str], dtype=object)
            )

            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        return responses
