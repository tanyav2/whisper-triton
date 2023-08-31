from transformers import (
    WhisperFeatureExtractor,
)
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-base"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            data = pb_utils.get_input_tensor_by_name(request, "audio").as_numpy()

            # Inference
            inputs = self.feature_extractor(
                data, sampling_rate=16000, return_tensors="pt"
            ).input_features

            inference_tensor = pb_utils.Tensor(
                "input_features", np.array(inputs)
            )

            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[inference_tensor]
            )
            responses.append(inference_response)
        return responses
