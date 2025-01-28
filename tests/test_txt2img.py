import requests
import io
from PIL import Image

# SCHEDULER OPTIONS:
# Euler
# Euler Karras
# Euler Ancestral
# Heun
# Heun Karras
# DPM++ 2M
# DPM++ 2M Karras
# DPM++ 2M SDE
# DPM++ 2M SDE Karras
# K DPM 2
# K DPM 2 Karras
# K DPM 2 Ancestral
# UniPC 2M
# UniPC 2M Karras
# LMS
# LMS Karras

def test_text_to_image():
    url = "http://localhost:8080/text-to-image"

    # Example of supported payload parameters
    payload = {
        "model_name": "model name only (no extension)",
        "prompt": "",
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "guidance": 6,
        "seed": None,
        "scheduler_name": "DPM++ 2M SDE Karras",
        "loras": [
            { "name": "model name only (no extension)", "weight": 1.2 },
            # { "name": "add more", "weight": 1.5 },
        ]
    }
    
    try:
        print("Sending request...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("Success! Saving image...")
            image = Image.open(io.BytesIO(response.content))
            image.save("test_output.png")
            print("Image saved as 'test_output.png'")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_text_to_image()