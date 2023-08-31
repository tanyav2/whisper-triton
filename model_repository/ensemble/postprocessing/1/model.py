from transformers import WhisperTokenizer
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-base", language="English", task="transcribe"
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            predicted_ids = pb_utils.get_input_tensor_by_name(request, "predicted_ids").as_numpy()

            # Inference
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
