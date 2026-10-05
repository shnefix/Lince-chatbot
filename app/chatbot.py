import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.config import MODEL_NAME

class Chatbot:
    def __init__(self):
        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

        self.model.eval()

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful and friendly assistant.",
            }
        ]

        print("Model loaded!")

    def chat(self, user_message: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Make sure we are working with the actual input tensor.
        if hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
        else:
            input_ids = inputs

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
            )

        new_tokens = outputs[0][input_ids.shape[-1]:]

        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        self.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        return response
