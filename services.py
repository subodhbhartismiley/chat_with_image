
import base64
import uuid
from typing import List, Optional, Dict
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
import datetime
from datetime import timedelta, datetime
import os
import json
import logging
import asyncio


class MultimodalMemoryManager:
    _sessions: Dict[str, 'MultimodalSession'] = {}

    @classmethod
    def get_or_create_session(cls, session_id: Optional[str] = None):
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in cls._sessions:
            cls._sessions[session_id] = MultimodalSession(session_id)
        
        return cls._sessions[session_id]

class MultimodalSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.image_context: Dict[str, Dict] = {}

    def add_image(self, base64_image: str, metadata: Optional[Dict] = None):
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        self.image_context[image_id] = {
            "base64": base64_image,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        return image_id

    def add_human_message(self, message: str):
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        self.memory.chat_memory.add_ai_message(message)

    def get_context(self):
        # Retrieve chat history and image context
        history = self.memory.chat_memory.messages
        images = [img_data['base64'] for img_data in self.image_context.values()]
        
        return {
            "history": history,
            "images": images,
            "image_context": self.image_context
        }

class MultimodalChatService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        # Correctly initialize the logger
        self.logger = LoggingManager()

    def process_chat(self, session_id: str, query: str, images: Optional[List[str]] = None):
        try:
            # Retrieve or create session
            session = MultimodalMemoryManager.get_or_create_session(session_id)

            # Process images if provided
            processed_images = []
            if images:
                for img in images:
                    # Add image to session context
                    image_id = session.add_image(img)
                    processed_images.append(img)

            # Prepare context for GPT-4o
            context = session.get_context()

            # # Construct detailed system prompt with context awareness
            # system_prompt = f"""
            # You are an advanced multimodal AI assistant with persistent context.
            # Your role is Biology, Physics, Chemistry and Mathematics Expert and advanced AI Assistant specialized in providing detailed descriptions of images. Upon receiving an image from a user, you conduct an exhaustive pixel-by-pixel analysis to ensure no detail is missed. Your comprehensive description includes:

            # - **Components**: Identification and description of every element within the image.
            # - **Size and Dimensions**: Precise measurements of each component.
            # - **Color**: Detailed color analysis, including specific shades and gradients.
            # - **Position**: Exact coordinates and spatial relationships between elements.
            # - **Height and Weight**: Physical attributes where applicable.
            # - **Probable Age and Gender**: Estimations based on visual cues and context.

            # In addition to these visual details, you also capture:

            # - **Text**: All text present in the image, including fonts and styles.
            # - **Mathematical Expressions and Equations**: Detailed representation and interpretation.
            # - **Graphs and Charts**: Description of curves, axes matching to the section and data points.
            # - **Symbols and Icons**: Identification and explanation of any symbolic elements and their correct spatials.
            # - **locations**: locations of each sections are very important, please ensure to use components locations properly and accurately. 

            # Furthermore, you integrate relevant theoretical and practical knowledge to provide context and enhance understanding. This includes:

            # - **Historical Context**: Background information related to the image's content.
            # - **Scientific Explanations**: Relevant theories and principles.
            # - **Practical Applications**: Real-world uses and implications.

            # Your generated text will be used as a context for a Multiturn and Multiple Image Question-Answering system, enabling precise and effective querying of the uploaded image. So, please ensure that no detail is overlooked.
 

            # Current Session Details:
            # - Session ID: {session_id}
            # - Total Images in Context: {len(context['images'])}
            
            # Context Preservation Guidelines:
            # - Always reference previously uploaded images
            # - Maintain conversation continuity
            # - Provide contextually rich responses
            # - If no new images are provided, use existing image context
            # """


            system_prompt = f"""
You are an advanced multimodal AI assistant with persistent context.
Your role is Biology, Physics, Chemistry and Mathematics Expert advanced AI Assistant specialized in providing detailed descriptions of images.
Follow below:

ADVANCED SPATIAL-AWARE MULTIMODAL IMAGE ANALYSIS PROTOCOL

PRIMARY OBJECTIVE:
Conduct hyper-precise, spatially-accurate image analysis with zero hallucination and maximum factual representation.

SPATIAL ANALYSIS FRAMEWORK:

1. COORDINATE-PRECISE COMPONENT MAPPING
- Implement absolute coordinate system for image analysis
- Use normalized coordinate space [0,1] for universal scaling
- Provide exact X, Y coordinates for each detected component
- Calculate precise relative and absolute positioning

2. SPATIAL RELATIONSHIP QUANTIFICATION
- Measure inter-object distances with sub-pixel accuracy
- Calculate angle and orientation between components
- Determine occlusion and overlap percentages
- Generate spatial relationship matrix
- Analysis with step by step reasoning

3. DIMENSIONAL PRECISION PROTOCOL
- Measurement Units: Pixels, Millimeters, Percentage of Image
- Provide confidence interval for measurements
- Use computer vision algorithms for precise sizing
- Compare against reference scales if available

4. LOCATION ACCURACY GUIDELINES
- Divide image into grid quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
- Assign precise quadrant location for each component
- Calculate percentage of component within each quadrant
- Provide exact pixel range for component boundaries
- Also provide location for each symbols, indicative symbols of components with step by step reasoning 

5. SPATIAL CONTEXT ANNOTATION
- Contextual Location Description
  * Absolute Position
  * Relative Position
  * Quadrant Location
  * Proximity to Image Edges
- Orientation Metrics
  * Rotation Angle
  * Horizontal/Vertical Alignment

6. CONFIDENCE AND VERIFICATION SYSTEM
- Confidence Levels for Spatial Measurements:
  * High Confidence (90-100%): Direct Pixel Measurement
  * Moderate Confidence (70-89%): Algorithmic Estimation
  * Low Confidence (50-69%): Approximation
  * Very Low Confidence (<50%): Requires Additional Verification

7. MULTI-DIMENSIONAL ANALYSIS PROTOCOL
- Spatial Analysis Dimensions:
  * 2D Coordinate Mapping
  * Depth Estimation
  * Relative Scale
  * Geometric Relationships

8. LOCATION FOR SELF CHECKING COMPONENTS SPATIALS WITH PROPER REASONING (say pixel-by-pixel components locations in image)
'''
    "component_name":
        "absolute_coordinates": 
                                "x" in float,
                                "y" in float
        
        "relative_position":
                                "quadrant" in string, 
                                "percentage_in_quadrant" in float
        
        "boundary_pixels": 
                                "top" in integer,
                                "bottom" in integer,
                                "left" in integer,
                                "right" in integer
    
        "confidence_level": in float
''' 

9. BE AWARE OF
- Current Session Details:
  * Session ID: {session_id}
  * Total Images in Context: {len(context['images'])}
  * By Default Refer Current Image otherwise stated by the user 

- Context Preservation Guidelines:
  * Always reference previously uploaded images
  * Maintain conversation continuity
  * Provide contextually rich responses
  * If no new images are provided, use existing image context
"""

            # Construct multimodal messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add chat history
            for msg in context['history']:
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        messages.append({"role": "assistant", "content": msg.content})

            # Add current query and images
            current_message = {"role": "user", "content": []}
            base_text = f"""        
                            - 1. locations of each component or section in the image are very important, please ensure to use components locations properly.
                            - 2. Please use location with reference to the right or left side of the real Object.
                            - 3. Please ensure components and their locations in the image and corresponding description matching correctly.
                            - 4. Please consider detailed description properly step by step with proper reasoning about all components in the image accordingly.
                            - 5. Please use the correct terminology to describe the image components.
                            - 6. Please ensure the description is accurate and precise.
                            - 7. Please use the correct format to describe the image components.
                            - 8. Please ensure the description is easy to understand.
                            - 9. Please use the description and image analysis as context for below.
                            - 10. Please ensure each components and its relations with all other components and spatials, indications, sybols are properly aligned with image pixel-by-pixel context with proper reasoning.
                            - 11. But please do not include image description in response (unless asked in focus query) because it costs extra money.
                            - 12. FOCUS QUERY: Please {query}, reward 1000$ will be given for the correct spatials, logics, flow and correct response.
                            Please refer all locations in two ways or atleast first way:
                            1st way - with reference to the image frame.
                            2nd way - with reference to the real Object (Object which is present in the image)
                            (say you are processing an image of the human heart then the position or locations of the components you refer in the heart should be the left side or right side of the human body. ).
                            Let's think step by step.
                            """
            current_message["content"].append({"type": "text", "text": base_text})

            # Add images if present
            if processed_images or context['images']:
                image_list = processed_images or context['images']
                for base64_img in image_list:
                    current_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    })

            messages.append(current_message)

            # Generate streaming response
            stream = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=16000,
                top_p=0.3,
                temperature=0.15,
                stream=True
            )

            # Accumulate full response for memory
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content


            # Log the interaction with comprehensive error handling
            try:
                log_file = self.logger.log_chat_interaction(
                    session_id=session_id, 
                    request={
                        "query": query,
                        "images": images
                    },
                    response=full_response,
                    images=images
                )
                logging.info(f"Interaction logged: {log_file}")
            except Exception as logging_error:
                logging.error(f"Detailed Logging Failed: {logging_error}")
                # Optional: Log with more context
                logging.error(f"Logging Context - Session ID: {session_id}, Query Length: {len(query)}")

            # Add response to session memory
            session.add_human_message(query)
            session.add_ai_message(full_response)

    
        except Exception as e:
            # Comprehensive error logging
            logging.error(f"Chat Processing Error: {e}", exc_info=True)
            # Optionally, you can yield an error message or re-raise
            yield f"An error occurred: {str(e)}"



    @classmethod
    def get_session_images(cls, session_id: str):
        """
        Retrieve all images for a specific session
        """
        session = MultimodalMemoryManager.get_or_create_session(session_id)
        context = session.get_context()
        return list(context['image_context'].keys())









class LoggingManager:
    def __init__(self, log_dir='logs'):
        # Create logs directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_chat_interaction(self, session_id: str, request: Dict, response: str, images: List[str] = None):
        """
        Log each chat interaction with comprehensive details
        """
        # Generate unique log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(
            self.log_dir, 
            f"session_{session_id}_{timestamp}.json"
        )

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "request": {
                "query": request.get('query', ''),
                "image_count": len(images) if images else 0
            },
            "response": {
                "length": len(response),
                "content": response
            },
            "metadata": {
                "images": [
                    {
                        "size": len(img),
                        "type": self._detect_image_type(img)
                    } for img in (images or [])
                ]
            }

        }

        # Write log entry
        with open(log_filename, 'w', encoding='utf-8') as log_file:
            json.dump(log_entry, log_file, indent=4, ensure_ascii=False)

        return log_filename
    


    def _detect_image_type(self, base64_image: str) -> str:
        """
        Detect image type from base64 encoded string
        """
        try:
            # Check image header
            if base64_image.startswith('/9j/'):  # JPEG
                return 'jpeg'
            elif base64_image.startswith('iVBORw0'):  # PNG
                return 'png'
            elif base64_image.startswith('R0lGOD'):  # GIF
                return 'gif'
            else:
                return 'unknown'
        except Exception:
            return 'unidentified'



