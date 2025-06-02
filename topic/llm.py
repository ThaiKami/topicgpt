from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List


class LabelModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        promp_path: str = "prompt/news_label_prompt.txt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.device = device
        try:
            with open(promp_path) as f:
                self.prompt = f.read()
        except Exception as e:
            print("Exception in loading prompt, will use default prompt instead")
            self.prompt = """You will receive a few online artcicles.
Your job is to come up with a one-sentence summary of all those documents 

[Instructions]
Write a summary from the documents. 
- The summary must be in Vietnamese
- Keep it short

[Document]
{Document}

Your response:"""
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)

    def generate(self, documents=List[str]):
        # prepare the model input
        documents_str = "\n".join(documents)
        prompt = self.prompt.format(Document=documents_str)
        messages = [{"role": "user", "content": prompt + " /no_think"}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # print(repr(text))
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        return thinking_content, content


if __name__ == "__main__":
    model = LabelModel(
        model_name="Qwen/Qwen3-0.6B",
        promp_path="prompt/news_label_prompt.txt",
        device="cuda",
    )

    list_of_doc = [
        "Không khí lạnh mạnh tràn về, miền Bắc rét đậm kèm mưa liên tiếp",
        "Thời tiết ngày 9/2: Bắc Bộ, Trung Bộ rét đậm, rét hại đến ngày 10/2",
        "Không khí lạnh mạnh lại tràn về, miền Bắc rét đậm kèm mưa liên tiếp",
        "Dự báo thời tiết ngày 26-02: Bắc Bộ rét đậm rét hại, Nam Bộ mưa to cục bộ",
    ]
    thinking, content = model.generate(documents=list_of_doc)

    print(thinking)
    print(content)
