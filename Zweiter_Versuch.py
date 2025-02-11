import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='mps')
print("AutoModel")

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()
print("AutoTokenizer")


Dateien=os.listdir("Pics")
Dateien = ["Pics/"+i for i in Dateien]

image = Image.open(Dateien[0]).convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print("res",res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    stream=True
)

print("------------")
generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')