# SIMPLE SDXL API USING DIFFUSERS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, memory-efficient API for Stable Diffusion XL operations. Built for production use with zero file storage and automatic resource cleanup.

## Features

- 🚀 Memory-efficient operation with zero file storage
- 🎨 Support for text-to-image, image-to-image, and inpainting
- 🔧 Multiple scheduler options
- 📦 LoRA model integration
- 🔄 Automatic device selection (CUDA/MPS/CPU)
- 🧹 Aggressive memory cleanup
- 🛡️ Production-ready error handling

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or Apple Silicon
- 16GB+ RAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sdxl-api.git
cd sdxl-api
```

2. Install dependencies:

For standard installations:
```bash
pip install -r requirements.txt
```

For MacOS Silicon (M1/M2) users:
```bash
# First install the nightly PyTorch build with MPS support
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu

# Then install remaining dependencies
pip install -r requirements.txt
```

3. Place your SDXL model files:
```
models/
├── main-models/
│   └── your-model.safetensors
└── lora-models/
    └── your-lora.safetensors
```

## Usage

Start the API server:
```bash
python -m src.api
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

#### Text to Image
```bash
curl -X POST "http://localhost:8000/text-to-image" \
     -H "Content-Type: application/json" \
     -d '{
           "model_name": "your-model",
           "prompt": "a photo of a cat",
           "width": 768,
           "height": 768,
           "steps": 30,
           "guidance": 7.0
         }'
```

#### Image to Image
```bash
curl -X POST "http://localhost:8000/image-to-image" \
     -F "image=@source.png" \
     -F 'request={
           "model_name": "your-model",
           "prompt": "a photo of a dog",
           "strength": 0.7
         }'
```

#### Inpainting
```bash
curl -X POST "http://localhost:8000/inpaint" \
     -F "image=@source.png" \
     -F "mask=@mask.png" \
     -F 'request={
           "model_name": "your-model",
           "prompt": "a photo of a bird"
         }'
```

### Configuration Options

#### Base Options
- `model_name`: Name of the model file (without extension)
- `prompt`: Text description of desired output
- `negative_prompt`: Text description of elements to avoid
- `steps`: Number of inference steps (default: 30)
- `guidance`: Classifier-free guidance scale (default: 7.0)
- `seed`: Random seed for reproducibility
- `scheduler_name`: Name of scheduler to use (default: "DPM++ 2M")

#### LoRA Options
```json
{
  "loras": [
    {
      "name": "your-lora",
      "weight": 0.7
    }
  ]
}
```

## Deployment

### Cloud Run Setup

1. Build the container:
```bash
docker build -t sdxl-api .
```

2. Configure memory limits:
```bash
gcloud run deploy sdxl-api \
  --image sdxl-api \
  --memory 16Gi \
  --cpu 4 \
  --timeout 300
```

### Memory Management

The API is designed for long-running deployments with:
- Zero file storage
- Automatic cache clearing
- Aggressive memory cleanup
- Resource monitoring

## Development

### Project Structure
```
sdxl-api/
├── src/
│   ├── __init__.py
│   ├── api.py        # FastAPI server implementation
│   └── inference.py  # SDXL inference engine
├── models/
│   ├── main-models/  # SDXL model files
│   └── lora-models/  # LoRA model files
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Stable Diffusion XL](https://stability.ai/stable-diffusion)
- [FastAPI](https://fastapi.tiangolo.com)
- [Diffusers](https://github.com/huggingface/diffusers)