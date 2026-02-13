import base64
import mimetypes
import os
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from core.config import AgentConfig

def _encode_image(image_path: str) -> str:
    """Encodes a local image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _get_mime_type(image_path: str) -> str:
    """Guesses the MIME type of the image."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        # Fallback for common extensions if mimetypes fails
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']: return 'image/jpeg'
        if ext == '.png': return 'image/png'
        if ext == '.webp': return 'image/webp'
        if ext == '.gif': return 'image/gif'
        return 'application/octet-stream'
    return mime_type

@tool
async def describe_image(image_path: str, prompt: Optional[str] = None) -> str:
    """
    Analyzes and describes an image file using the configured Vision Model.
    Args:
        image_path: Absolute or relative path to the image file.
        prompt: Optional specific question or instruction about the image.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    try:
        # 1. Load Config (to get API keys and provider)
        config = AgentConfig()
        
        # 2. Prepare Image
        mime_type = _get_mime_type(image_path)
        base64_image = _encode_image(image_path)
        
        if not prompt:
            prompt = "Describe this image in detail. Focus on the main subjects, setting, and any text visible."

        # 3. Construct Message
        # LangChain standard format for multimodal inputs
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        )

        # 4. Initialize LLM (fresh instance to avoid dependency injection issues in tools)
        if config.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=config.gemini_model,
                google_api_key=config.gemini_api_key.get_secret_value(),
                temperature=0.2
            )
        elif config.provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=config.openai_model,
                api_key=config.openai_api_key.get_secret_value(),
                temperature=0.2
            )
        else:
            return f"Error: Provider '{config.provider}' does not support vision tools yet."

        # 5. Invoke
        response = await llm.ainvoke([message])
        return response.content

    except Exception as e:
        return f"Error processing image: {str(e)}"
