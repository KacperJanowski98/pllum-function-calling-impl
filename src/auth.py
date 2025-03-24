"""Authentication utilities for Hugging Face."""

import os
from pathlib import Path

import dotenv
from huggingface_hub import login


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        dotenv.load_dotenv(env_path)
    else:
        print("Warning: .env file not found. Please create one from .env.example")


def get_hf_token():
    """Get Hugging Face token from environment variables."""
    load_environment()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please create a .env file with your Hugging Face token."
        )
    return token


def login_to_huggingface():
    """Login to Hugging Face using token from environment variables."""
    token = get_hf_token()
    login(token=token)
    print("Successfully logged in to Hugging Face")
