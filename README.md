# SDXL API
### Super Simple Implementation of the SDXL Pipelines from Diffusers as an API.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, memory-efficient API for Stable Diffusion XL operations. Built for production use with zero file storage and automatic resource cleanup.

run.sh and runMac.sh are both for installation and running the app.
runMac.sh installs a nightly version of PyTorch for better FP16 support on MPS.

## Features

- 🚀 Memory-efficient operation with zero file storage
- 🎨 Support for text-to-image, image-to-image, and inpainting
- 🔧 Multiple scheduler options
- 📦 LoRA model integration
- 🔄 Automatic device selection (CUDA/MPS/CPU)
- 🧹 Aggressive memory cleanup
- 🛡️ Production-ready error handling

### For MacOS Silicon (M1/M2) users:
```bash
# As of the time of posting this README, Mac Silicone users should use this Nightly version of Torch for FP16 support on MPS. This is already included in runMac.sh, so just use bash to run that script.
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```
The server will start on `http://localhost:8000` by default.

## Memory Management

The API is designed for long-running deployments with:
- Zero file storage
- Automatic cache clearing
- Aggressive memory cleanup
- Resource monitoring

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Stable Diffusion XL](https://stability.ai/stable-diffusion)
- [FastAPI](https://fastapi.tiangolo.com)
- [Diffusers](https://github.com/huggingface/diffusers)