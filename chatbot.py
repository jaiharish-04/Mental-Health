# make these installations
"""
pip install transformers peft
"""

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch


class CBflan_t5:
    def __init__(self, config_path):
        self.model_name = 'google/flan-t5-base'

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.peft_model = PeftModel.from_pretrained(self.model, config_path).to("cpu")

    def reply(self, user_input):
        prompt = f"""generate an answer for this question:

        {user_input}

        \n Answer:
        """

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

        peft_model_outputs = self.peft_model.generate(input_ids=input_ids,
                                                      generation_config=GenerationConfig(min_new_tokes=20, do_sample=True,
                                                                                         nbeams=4, temperature=1.5))

        peft_model_text_output = self.tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

        return peft_model_text_output


# add these lines in your backend's logic (save the entire file as chatbot.py
# from chatbot import CBflan_t5

path_to_config_in_your_machine = ('/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/backend'
                                  '/peftcheckpoint_mark_test')
o = CBflan_t5(path_to_config_in_your_machine)
print(o.reply('I am having a bad headache, give me a solution'))
