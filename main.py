import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from dotenv import load_dotenv
from datetime import datetime
import json

import logging
from logging.handlers import RotatingFileHandler

# Import local modules
from models import ImageUpload, ChatRequest
from services import MultimodalChatService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Multimodal Chat System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat service
chat_service = MultimodalChatService(api_key=os.getenv("OPENAI_API_KEY"))




def validate_and_convert_image(image_input):
    """
    Validate and convert image to base64
    Supports: 
    - Local file paths
    - Base64 strings
    - URL (optional)
    """
    try:
        # If it's a local file path
        if os.path.exists(image_input):
            # Check file size
            file_size = os.path.getsize(image_input)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Image file too large")
            
            # Check file type
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif']
            if not any(image_input.lower().endswith(ext) for ext in allowed_extensions):
                raise ValueError("Unsupported image type")
            
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # If it's already a base64 string
        elif image_input.startswith(('data:image', 'base64,')):
            return image_input
        
        # Optional: Add URL handling
        # elif image_input.startswith(('http://', 'https://')):
        #     response = requests.get(image_input)
        #     return base64.b64encode(response.content).decode('utf-8')
        
        else:
            raise ValueError("Invalid image input")
    
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# # Update chat_endpoint to use this function
# processed_images = [validate_and_convert_image(img) for img in request.images]


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), session_id: str = None):
    try:
        # Read image file
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        return {
            "message": "Image uploaded successfully",
            "session_id": session_id or str(uuid.uuid4()),
            "image": base64_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def convert_to_base64(image_input):
    """
    Convert image to base64 string
    Supports:
    - Local file paths
    - Base64 strings
    """
    try:
        # If it's a local file path
        if os.path.exists(image_input):
            # Validate file size
            file_size = os.path.getsize(image_input)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Image file too large")
            
            # Validate file type
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.dcm', '.dicom']
            if not any(image_input.lower().endswith(ext) for ext in allowed_extensions):
                raise ValueError("Unsupported image type")
            
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # If it's already a base64 string
        elif image_input.startswith(('data:image', 'base64,')):
            return image_input.split(',')[-1]
        
        else:
            raise ValueError("Invalid image input")
    
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")




import asyncio

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # If no session_id is provided, generate a new one
        session_id = request.session_id or str(uuid.uuid4())

        # Convert images to base64
        processed_images = []
        if request.images:
            processed_images = [
                convert_to_base64(img) for img in request.images
            ]

        # Create generator for streaming response
        response_content = ""
        def generate():
            nonlocal response_content
            for chunk in chat_service.process_chat(
                session_id=session_id, 
                query=request.query, 
                images=processed_images
            ):
                response_content += chunk
                # yield json.dumps({"data": chunk.encode('utf-8')})
                # yield f"data: {chunk.encode('utf-8')}\n\n"
                yield f"data: {chunk}\n\n"


                # await asyncio.sleep(0.0001)

        # Return streaming response with session ID
        return StreamingResponse(
            generate(), 
            # media_type="text/plain", 
            media_type="text/event-stream",

            headers={
                "X-Session-ID": session_id
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# @app.get("/session/{session_id}/images")
# async def get_session_images(session_id: str):
#     try:
#         # Retrieve image IDs for the session
#         image_ids = MultimodalChatService.get_session_images(session_id)
        
#         return {
#             "session_id": session_id,
#             "image_count": len(image_ids),
#             "image_ids": image_ids
#         }
#     except Exception as e:
#         raise HTTPException(status_code=404, detail="Session not found")

# @app.get("/session/{session_id}/image/{image_id}")
# async def get_session_image(session_id: str, image_id: str):
#     try:
#         # Retrieve the session
#         session = MultimodalMemoryManager.get_or_create_session(session_id)
        
#         # Get specific image from context
#         if image_id in session.image_context:
#             image_data = session.image_context[image_id]
#             return {
#                 "session_id": session_id,
#                 "image_id": image_id,
#                 "base64": image_data['base64'],
#                 "metadata": image_data['metadata'],
#                 "timestamp": image_data['timestamp']
#             }
#         else:
#             raise HTTPException(status_code=404, detail="Image not found in session")
#     except Exception as e:
#         raise HTTPException(status_code=404, detail="Session or image not found")



@app.get("/logs")
async def get_log_files():
    """
    Retrieve list of log files
    """
    try:
        log_files = [
            f for f in os.listdir('logs') 
            if f.endswith('.json') and f.startswith('session_')
        ]
        
        # Get file details
        log_details = []
        for filename in log_files:
            full_path = os.path.join('logs', filename)
            file_stats = os.stat(full_path)
            log_details.append({
                "filename": filename,
                "size": file_stats.st_size,
                "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
            })
        
        return {
            "total_logs": len(log_files),
            "logs": log_details
        }
    except Exception as e:
        logging.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve log files")



@app.get("/logs/{filename}")
async def get_log_file(filename: str):
    """
    Retrieve specific log file
    """
    try:
        file_path = os.path.join('logs', filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        with open(file_path, 'r') as log_file:
            log_content = json.load(log_file)
        
        return log_content
    except Exception as e:
        logging.error(f"Error reading log file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not read log file")




# Configure logging
def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler for detailed logs
            RotatingFileHandler(
                'logs/application.log', 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            ),
            # Console handler
            logging.StreamHandler()
        ]
    )

    # Add specific loggers
    chat_logger = logging.getLogger('MultimodalChatService')
    chat_logger.setLevel(logging.INFO)


# Call setup_logging during app initialization
setup_logging()





# import logging
# from logging.handlers import RotatingFileHandler
# import os

# def setup_advanced_logging():
#     # Ensure logs directory exists
#     os.makedirs('logs', exist_ok=True)
    
#     # Configure logging with more detailed settings
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             # Rotating file handler for application logs
#             RotatingFileHandler(
#                 'logs/application.log', 
#                 maxBytes=10*1024*1024,  # 10 MB
#                 backupCount=5
#             ),
#             # Error-specific log file
#             RotatingFileHandler(
#                 'logs/errors.log', 
#                 maxBytes=5*1024*1024,  # 5 MB
#                 backupCount=3,
#                 level=logging.ERROR
#             ),
#             # Console handler
#             logging.StreamHandler()
#         ]
#     )

#     # Set specific loggers
#     logging.getLogger('openai').setLevel(logging.WARNING)
#     logging.getLogger('httpx').setLevel(logging.WARNING)

# # Call this during application startup
# setup_advanced_logging()




# import traceback
# def log_exception(e: Exception, context: Dict = None):
#     """
#     Comprehensive exception logging
#     """
#     error_details = {
#         "error_type": type(e).__name__,
#         "error_message": str(e),
#         "traceback": traceback.format_exc()
#     }
    
#     if context:
#         error_details["context"] = context
    
#     logging.error(json.dumps(error_details, indent=2))


# def log_dependencies():
#     """
#     Log important dependency versions
#     """
#     import sys
#     import openai
#     import httpx

#     dependency_info = {
#         "python_version": sys.version,
#         "openai_version": openai.__version__,
#         "httpx_version": httpx.__version__
#     }
    
#     logging.info(f"Dependency Versions: {json.dumps(dependency_info, indent=2)}")

# # Call during application initialization
# log_dependencies()



# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)