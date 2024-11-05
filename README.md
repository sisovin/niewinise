
# Clean-UI for Multi-Modal Vision Models

This project offers a user-friendly interface for working with the **Llama-3.2:3b** and **Llama-3.2:1b** models.

In this case, both the **Llama-3.2:3b** and **Llama-3.2:1b** models need 12GB of VRAM to run. Using "API_BASE"="'http://localhost:11434/api/generate'" when you download Llama3.2:1b or Llama3.2:3b from Ollama installed your local computer of Windows 11. 

The model selection is done via the command line:

## Installation

To set up and run this project on your local machine, follow the steps below:

### 1. Clone the Repository

Copy the repository to a convenient location on your computer:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

Inside the cloned repository, create a virtual environment using the following command:

```bash
python -m venv venv-ui
```

### 3. Activate the Virtual Environment

Activate the virtual environment using:

  ```bash
  .\venv-ui\Scripts\activate
  ```

### 4. Install Dependencies

After activating the virtual environment, install the necessary dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
Install Torch and TorchVision using separate commands:
```bash
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
and

```bash
pip install torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
## Usage

To start the UI, you can either:

- Use the **run.bat** script (Windows only)
  
  Simply double-click on `run.bat` or

  1. Activate the virtual environment:
     
     - Windows:
       ```bash
       .\venv-ui\Scripts\activate
       ```

  2. Run the Python script:
  
     ```bash
     python clean-ui.py
     ```

## Features

- Upload an image and enter a prompt to generate an image description.
- Adjustable parameters such as temperature, top-k, and top-p for more control over the generated text.
- Chatbot history to display prompt-response interactions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

```python

import gradio as gr
import torch
import os
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or adjust size as needed

# Model selection menu in terminal
print("Select a model to load:")
print("1. Llama-3.2-11B-Vision-Instruct-bnb-4bit")
print("2. Molmo-7B-D-bnb-4bit")
model_choice = input("Enter the number of the model you want to use: ")

if model_choice == "1":
    model_id = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

elif model_choice == "2":
    model_id = "cyan2k/molmo-7B-D-bnb-4bit"
    arguments = {"device_map": "auto", "torch_dtype": "auto", "trust_remote_code": True}
    model = AutoModelForCausalLM.from_pretrained(model_id, **arguments)
    processor = AutoProcessor.from_pretrained(model_id, **arguments)

else:
    raise ValueError("Invalid model choice. Please enter 1 or 2.")

# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

# Function to process the image and generate a description
def describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens, history):
    # Resize image if necessary
    image = image.resize(MAX_IMAGE_SIZE)

    # Initialize cleaned_output variable
    cleaned_output = ""

    # Prepare prompt with user input based on selected model
    if model_choice == "1":  # Llama Model
        prompt = f"<|image|><|begin_of_text|>{user_prompt} Answer:"
        # Preprocess the image and prompt
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        # Generate output with model
        output = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Decode the raw output
        raw_output = processor.decode(output[0])
        
        # Clean up the output to remove system tokens
        cleaned_output = raw_output.replace("<|image|><|begin_of_text|>", "").strip().replace(" Answer:", "")

    elif model_choice == "2":  # Molmo Model
        # Prepare inputs for Molmo model
        inputs = processor.process(images=[image], text=user_prompt)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Generate output with model, applying the parameters for temperature, top_k, top_p, and max_tokens
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_strings="<|endoftext|>",
                do_sample=True
            ),
            tokenizer=processor.tokenizer,
        )

        # Extract generated tokens and decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        cleaned_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Ensure the prompt is not repeated in the output
    if cleaned_output.startswith(user_prompt):
        cleaned_output = cleaned_output[len(user_prompt):].strip()
        
    # Append the new conversation to the history
    history.append((user_prompt, cleaned_output))

    return history

# Function to clear the chat history
def clear_chat():
    return []

# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    Clean-UI
    </h1>
    """)
        with gr.Row():
            # Left column with image and parameter inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image", 
                    type="pil", 
                    image_mode="RGB", 
                    height=512,  # Set the height
                    width=512   # Set the width
                )

                # Parameter sliders
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.6, step=0.1, interactive=True)
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=50, step=1, interactive=True)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50, interactive=True)

            # Right column with the chat interface
            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512)

                # User input box for prompt
                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                # Generate and Clear buttons
                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

                # Define the action for the generate button
                generate_button.click(
                    fn=describe_image, 
                    inputs=[image_input, user_prompt, temperature, top_k, top_p, max_tokens, chat_history],
                    outputs=[chat_history]
                )

                # Define the action for the clear button
                clear_button.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[chat_history]
                )

    return demo

# Launch the interface
demo = gradio_interface()
demo.launch()

```

```python
import gradio as gr
import os
import json
import requests

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or adjust size as needed

# Model selection menu in terminal
print("Select a model to load:")
print("1. llama3.2:3b")
print("2. llama3.2:1b")
model_choice = input("Enter the number of the model you want to use: ")

if model_choice == "1":
    model_id = "llama3.2:3b"
elif model_choice == "2":
    model_id = "llama3.2:1b"
else:
    raise ValueError("Invalid model choice. Please enter 1 or 2.")

api_base = 'http://localhost:11434/api/generate'

# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048

# Function to generate a response based on user input
def generate_response(user_prompt, temperature, top_k, top_p, max_tokens, history):
    # Prepare the prompt
    prompt = f"{user_prompt} Answer:"

    # Prepare the data for the API request
    data = {
        'prompt': json.dumps(prompt),  # Convert prompt to JSON string
        'model': model_id,
        'provider': 'ollama',
        'title': model_id
    }

    # Make the API request
    response = requests.post(f"{api_base}", json=data)

    # Check for 404 error
    if response.status_code == 404:
        print("Error: API endpoint not found.")
        return history

    # Print the response text for debugging
    print("API Response Text:", response.text)

    # Process each JSON object in the response
    cleaned_output = ""
    for line in response.text.splitlines():
        try:
            response_data = json.loads(line)
            cleaned_output += response_data.get('response', '')
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            continue

    # Ensure the prompt is not repeated in the output
    if cleaned_output.startswith(user_prompt):
        cleaned_output = cleaned_output[len(user_prompt):].strip()

    # Append the new conversation to the history
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": cleaned_output})

    return history

# Function to clear the chat history
def clear_chat():
    return []

# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    AI Chatbot
    </h1>
    """)
        with gr.Row():
            # Left column with parameter inputs
            with gr.Column(scale=1):
                # Parameter sliders
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=0.6, step=0.1, interactive=True)
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=50, step=1, interactive=True)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50, interactive=True)

            # Right column with the chat interface
            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512, type='messages')

                # User input box for prompt
                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                # Generate and Clear buttons
                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

                # Define the action for the generate button
                generate_button.click(
                    fn=generate_response, 
                    inputs=[user_prompt, temperature, top_k, top_p, max_tokens, chat_history],
                    outputs=[chat_history]
                )

                # Define the action for the clear button
                clear_button.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[chat_history]
                )

    return demo

# Launch the interface
demo = gradio_interface()
demo.launch(share=True)

```


To record the chat history between the user prompt and the chatbot using the 

Logger

 class, you can add logging statements in the 

generate_response

 function. This will log each user prompt and the corresponding chatbot response.

### Updated 

generate_response

 function
Add logging statements to record the chat history.

#### 

niewinise.py


```python
import gradio as gr
import os
import json
import requests
from functions.logger import Logger

# Set memory management for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or adjust size as needed

# Model selection menu in terminal
print("Select a model to load:")
print("1. llama3.2:3b")
print("2. llama3.2:1b")
model_choice = input("Enter the number of the model you want to use: ")

if model_choice == "1":
    model_id = "llama3.2:3b"
elif model_choice == "2":
    model_id = "llama3.2:1b"
else:
    raise ValueError("Invalid model choice. Please enter 1 or 2.")

api_base = 'http://localhost:11434/api/generate'

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO)

# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048

# Function to generate a response based on user input
def generate_response(user_prompt, top_k, top_p, max_tokens, history):
    # Log the user prompt
    logger.info(f"User Prompt

:

 {user_prompt}")

    # Prepare the prompt
    prompt = f"{user_prompt} Answer:"

    # Prepare the data for the API request
    data = {
        'prompt': json.dumps(prompt),  # Convert prompt to JSON string
        'model': model_id,
        'provider': 'ollama',
        'title': model_id
    }

    # Make the API request
    response = requests.post(f"{api_base}", json=data)

    # Check for 404 error
    if response.status_code == 404:
        logger.error("Error: API endpoint not found.")
        return history

    # Print the response text for debugging
    logger.info(f"API Response Text: {response.text}")

    # Process each JSON object in the response
    cleaned_output = ""
    for line in response.text.splitlines():
        try:
            response_data = json.loads(line)
            cleaned_output += response_data.get('response', '')
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            continue

    # Ensure the prompt is not repeated in the output
    if cleaned_output.startswith(user_prompt):
        cleaned_output = cleaned_output[len(user_prompt):].strip()

    # Log the chatbot response
    logger.info(f"Chatbot Response: {cleaned_output}")

    # Append the new conversation to the history
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": cleaned_output})

    return history

# Function to clear the chat history
def clear_chat():
    return []

# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    AI Chatbot
    </h1>
    """)
        with gr.Row():
            # Right column with the chat interface
            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512, type='messages')

                # User input box for prompt
                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                # Parameter sliders
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=50, step=1, interactive=True)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50, interactive=True)

                # Generate and Clear buttons
                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

                # Define the action for the generate button
                generate_button.click(
                    fn=generate_response, 
                    inputs=[user_prompt, top_k, top_p, max_tokens, chat_history],
                    outputs=[chat_history]
                )

                # Define the action for the clear button
                clear_button.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[chat_history]
                )

    return demo

# Launch the interface
demo = gradio_interface()
demo.launch(share=True)
```

### Explanation
1. **Logger Initialization**: The 

Logger

 is initialized at the beginning of the script.
2. **Logging User Prompt**: The user prompt is logged using 

logger.info

.
3. **Logging API Response**: The API response text is logged for debugging purposes.
4. **Logging Chatbot Response**: The cleaned chatbot response is logged using 

logger.info

.

This setup will log each user prompt and the corresponding chatbot response to the specified log file.

### Conclusion

The Clean-UI project provides a robust and user-friendly interface for interacting with the **Llama-3.2:3b** and **Llama-3.2:1b** models. By leveraging the power of these models, users can generate detailed image descriptions and engage in meaningful conversations with the AI chatbot. The project is designed to be easily set up and run on a local machine, with clear instructions provided for installation and usage. Additionally, the integration of a logging system ensures that all interactions are recorded, allowing for easy debugging and analysis of the chatbot's performance. This makes Clean-UI a valuable tool for developers and researchers working with multi-modal vision models.
