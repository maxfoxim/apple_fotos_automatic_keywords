import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image

# Load the model
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
#image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
image_file = "5.png"
image = [Image.open(image_file)] #can also be used with PIL.Image.Image objects
prompt = "Describe this image." #klapp
prompt = "Give me a list of objects in the image."
prompt = "Do you see a human in the image?"
prompt = "Give this image a long title in german!" #klappt
prompt = "Give this image ten keywords"



# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)
"""
print("processor, config, prompt")
print(processor, config, prompt)

print("formatted_prompt")
print(formatted_prompt)
"""

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=True)
print(output)