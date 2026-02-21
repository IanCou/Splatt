import os
import logging
import base64
from io import BytesIO
from typing import List
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from models import SceneQueryResponse

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def image_to_base64_url(img: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URL.

    Args:
        img: PIL Image object

    Returns:
        Base64-encoded data URL string (e.g., "data:image/png;base64,...")
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return f"data:image/png;base64,{img_base64}"


class GeminiService:
    """
    Service layer for all Gemini AI operations in Splatt.

    Encapsulates LangChain interactions and provides clean abstractions
    for multimodal video analysis and structured scene querying.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini service with LangChain.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Initialize LangChain ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )

        # Create structured output version for queries
        self.structured_llm = self.llm.with_structured_output(
            SceneQueryResponse,
            method="json_mode"
        )

        logger.info(f"GeminiService initialized with model: {model_name}")

    def analyze_construction_scene(self, keyframes: List[Image.Image]) -> str:
        """
        Analyze construction site video keyframes to generate scene summary.

        This is a multimodal operation that processes both text prompts
        and image data to understand the construction environment.

        Args:
            keyframes: List of PIL Image objects (typically 2-10 frames)

        Returns:
            Scene summary string describing objects, equipment, activities,
            and spatial relationships
        """
        if not keyframes:
            raise ValueError("At least one keyframe is required for analysis")

        logger.info(f"Analyzing {len(keyframes)} keyframes...")

        # Construct analysis prompt
        analysis_prompt = """
        This is a sequence of frames from a construction site 'helmet cam'.
        Analyze the scene and describe:
        1. Major objects/materials (e.g. lumber, concrete)
        2. Equipment present
        3. Activities being performed
        4. Spatial relationships

        Format as a concise summary for a 3D scene index.
        """

        # Build message content with text + images
        message_content = [{"type": "text", "text": analysis_prompt}]

        for img in keyframes:
            # Convert PIL Image to base64 data URL
            img_data_url = image_to_base64_url(img)
            message_content.append({
                "type": "image_url",
                "image_url": img_data_url
            })

        # Create HumanMessage with multimodal content
        message = HumanMessage(content=message_content)

        # Invoke LLM
        print(f"[GEMINI] Sending {len(keyframes)} keyframes to Gemini for analysis...")
        response = self.llm.invoke([message])

        print(f"[GEMINI] Analysis complete. Response length: {len(response.content)} characters")
        print(f"[GEMINI] Response preview: {response.content[:300]}...")
        logger.info("Scene analysis complete")
        return response.content

    def query_scene(self, user_query: str, scene_context: str) -> SceneQueryResponse:
        """
        Query construction site scenes with structured output.

        Uses Pydantic schema to ensure consistent, validated JSON responses
        containing analysis, location data, confidence, and worker information.

        Args:
            user_query: Natural language question from user
            scene_context: Combined scene summaries to query against

        Returns:
            SceneQueryResponse with validated structured data
        """
        logger.info(f"Processing query: {user_query[:50]}...")
        print(f"\n[GEMINI] Query scene called")
        print(f"[GEMINI] User query: {user_query}")
        print(f"[GEMINI] Scene context length: {len(scene_context)} characters")

        prompt = f"""
        Based on the following construction site scene analysis:
        {scene_context}

        User Query: {user_query}

        Respond with:
        1. 'analysis': A detailed natural language answer.
        2. 'location': Specific site location if mentioned (e.g. "South wall").
        3. 'coordinates': Mock coordinates if possible (e.g. "X:10, Y:20").
        4. 'confidence': 'High', 'Medium', or 'Low'.
        5. 'hotspots': A list of matching hotspot IDs (h1, h2, h3, h4, h5, h6) if applicable.
        6. 'worker': The name of a worker if seen or inferred.
        7. 'workerRole': Their role.
        """

        print(f"[GEMINI] Sending structured query to Gemini...")
        # Use structured output - automatically validates against Pydantic model
        response: SceneQueryResponse = self.structured_llm.invoke(prompt)

        print(f"[GEMINI] Structured response received:")
        print(f"  - Analysis: {response.analysis[:100] if response.analysis else 'None'}...")
        print(f"  - Confidence: {response.confidence}")
        print(f"  - Hotspots: {response.hotspots}")
        print(f"  - Location: {response.location}")
        print(f"  - Coordinates: {response.coordinates}")
        print(f"  - Worker: {response.worker}")
        print(f"  - Worker Role: {response.workerRole}")

        logger.info(f"Query complete with confidence: {response.confidence}")
        return response


# Singleton instance for application use
_gemini_service = None


def get_gemini_service() -> GeminiService:
    """
    Get or create singleton GeminiService instance.

    This pattern ensures we reuse the same LLM instance across requests,
    avoiding unnecessary re-initialization overhead.
    """
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
