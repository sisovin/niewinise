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
    logger.info(f"User Prompt: {user_prompt}")

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
    # logger.info(f"API Response Text: {response.text}")

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
    Niewinise Chatbot
    </h1>
    """)
        with gr.Row():
            # Right column with the chat interface
            with gr.Column(scale=2):
                # Parameter sliders
                top_k = gr.Slider(
                  label="Top-k", minimum=1, maximum=100, value=50, step=1, interactive=True)
                top_p = gr.Slider(
                  label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                  label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=100, step=50, interactive=True)
                            
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
                    inputs=[user_prompt, top_k, top_p, max_tokens, chat_history],
                    # inputs=[user_prompt, chat_history],
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