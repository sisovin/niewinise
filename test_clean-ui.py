import pytest
from PIL import Image
from io import BytesIO
from niewinise import gradio_interface, describe_image, clear_chat

@pytest.fixture
def demo():
    return gradio_interface()

def test_gradio_interface_initialization(demo):
    assert demo is not None

def test_describe_image():
    # Create a dummy image
    image = Image.new('RGB', (100, 100), color='red')
    user_prompt = "Describe this image"
    temperature = 0.6
    top_k = 50
    top_p = 0.9
    max_tokens = 100
    history = []

    result = describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens, history)
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], tuple)
    assert result[0][0] == user_prompt

def test_clear_chat():
    history = [("User prompt", "Response")]
    result = clear_chat()
    assert result == []