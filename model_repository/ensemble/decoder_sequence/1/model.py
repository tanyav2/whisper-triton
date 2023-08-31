import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            predicted_ids = np.array([[50258]])
            for i in range(100):
                encoder_hidden_states = pb_utils.get_input_tensor_by_name(request, "last_hidden_state")
                encoder_hidden_states_tens = pb_utils.Tensor(
                    "encoder_hidden_states", np.array(encoder_hidden_states)
                )
                input_ids_tens = pb_utils.Tensor(
                    "input_ids", np.array(predicted_ids)
                )
                decoder_outputs_req = pb_utils.InferenceRequest(
                    model_name="decoder_model",
                    requested_output_names=["logits"],
                    inputs=[encoder_hidden_states_tens, input_ids_tens]
                )
                logits_resp = decoder_outputs_req.exec()
                if logits_resp.has_error():
                    raise pb_utils.TritonModelException(
                        logits_resp.error().message()
                    )
                else:
                    logits = pb_utils.get_output_tensor_by_name(
                        logits_resp, "logits"
                    ).as_numpy()

                logits_of_last_token = logits[0][0, -1, :]
                next_token = np.argmax(logits_of_last_token)
                predicted_ids = np.append(predicted_ids, [[next_token]], axis=1)

            inference_tensor = pb_utils.Tensor(
                "predicted_ids", predicted_ids
            )

            # Create inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[inference_tensor]
            )
            responses.append(inference_response)
        return responses