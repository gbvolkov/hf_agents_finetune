from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch

import os

with open('hf_token.txt', 'r') as f:
    os.environ['HF_TOKEN'] = f.read()    
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

"""
model_name = "google/gemma-2-2b-it"
dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

def preprocess(sample):
      messages = sample["messages"]
      first_message = messages[0]

      # Instead of adding a system message, we merge the content into the first user message
      if first_message["role"] == "system":
          system_message_content = first_message["content"]
          # Merge system content with the first user message
          messages[1]["content"] = system_message_content + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n" + messages[1]["content"]
          # Remove the system message from the conversation
          messages.pop(0)

      return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.map(preprocess, remove_columns="messages")
dataset = dataset["train"].train_test_split(0.1)
print(dataset)
"""

username="gbv"# REPLCAE with your Hugging Face username
#output_dir = "gemma-2-2B-it-thinking-function_calling-V0" # The directory where the trained model checkpoints, logs, and other artifacts will be saved. It will also be the default name of the model when pushed to the hub if not redefined later.
output_dir = "SmolLM2-1.7B-Instruct-thinking-function_calling-V0" # The directory where the trained model checkpoints, logs, and other artifacts will be saved. It will also be the default name of the model when pushed to the hub if not redefined later.

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

peft_model_id = f"{username}/{output_dir}" # replace with your newly trained adapter
device = "auto"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             device_map="auto",
                                             )
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(device="cuda", dtype=torch.bfloat16)
model.eval()



#print(dataset["test"][8]["text"])

"""### Testing the model 🚀

In that case, we will take the start of one of the samples from the test set and hope that it will generate the expected output.

Since we want to test the function-calling capacities of our newly fine-tuned model, the input will be a user message with the available tools, a


### Disclaimer ⚠️

The dataset we’re using **does not contain sufficient training data** and is purely for **educational purposes**. As a result, **your trained model’s outputs may differ** from the examples shown in this course. **Don’t be discouraged** if your results vary—our primary goal here is to illustrate the core concepts rather than produce a fully optimized or production-ready model.

"""

#this prompt is a sub-sample of one of the test set examples. In this example we start the generation after the model generation starts.
prompt="""<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>
<start_of_turn>model
<think>"""

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {k: v.to("cuda") for k,v in inputs.items()}
outputs = model.generate(**inputs,
                         max_new_tokens=300,# Adapt as necessary
                         do_sample=True,
                         top_p=0.95,
                         temperature=0.01,
                         repetition_penalty=1.0,
                         eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))

"""## Congratulations
Congratulations on finishing this first Bonus Unit 🥳

You've just **mastered what Function-Calling is and how to fine-tune your model to do Function-Calling**!

If it's the first time you do this, it's normal that you're feeling puzzled. Take time to check the documentation and understand each part of the code and why we did it this way.

Also, don't hesitate to try to **fine-tune different models**. The **best way to learn is by trying.**

### Keep Learning, Stay Awesome 🤗
"""