from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
)
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-base"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base"
        )
        self.model.config.forced_decoder_ids = None
        self.tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-base", language="English", task="transcribe"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            data = pb_utils.get_input_tensor_by_name(request, "audio").as_numpy()

            # Inference
            inputs = self.feature_extractor(
                data, sampling_rate=16000, return_tensors="pt"
            ).input_features
            predicted_ids = self.model.generate(inputs)
            transcription = self.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            # Postprocessing
            transcription_str = transcription[0]
            output_tensor = pb_utils.Tensor(
                "transcription", np.array([transcription_str], dtype=np.object)
            )

            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
