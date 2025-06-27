# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import io
from PIL import Image
import numpy as np
import json
import logging
from typing import Dict, Any
from image_classifier import ImageClassifier
from ocr_validator import OCRValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="College ID Validator",
    description="AI-based system to detect fake, altered, or non-genuine student ID card uploads",
    version="1.0.0"
)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize components
image_classifier = ImageClassifier()
ocr_validator = OCRValidator(config)

class IDValidationRequest(BaseModel):
    user_id: str
    image_base64: str

class IDValidationResponse(BaseModel):
    user_id: str
    validation_score: float
    label: str
    status: str
    reason: str
    threshold: float

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data:image/jpeg;base64, prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def calculate_final_score(image_score: float, ocr_score: float, template_score: float) -> float:
    """Calculate weighted final validation score"""
    weights = {
        'image': 0.4,
        'ocr': 0.4,
        'template': 0.2
    }
    
    final_score = (
        image_score * weights['image'] +
        ocr_score * weights['ocr'] +
        template_score * weights['template']
    )
    
    return min(max(final_score, 0.0), 1.0)

def determine_label_and_status(score: float, ocr_confidence: float, threshold: float) -> tuple:
    """Determine label and status based on score and OCR confidence"""
    if score > 0.85 and ocr_confidence > 0.7:
        return "genuine", "approved", "High confidence genuine ID"
    elif (0.6 <= score <= 0.85) or (ocr_confidence < 0.7 and score > 0.5):
        return "suspicious", "manual_review", "Moderate confidence, requires manual review"
    else:
        if score < 0.3:
            return "fake", "rejected", "Low confidence - likely fake or manipulated"
        elif ocr_confidence < 0.3:
            return "fake", "rejected", "OCR validation failed - poor quality or fake"
        else:
            return "fake", "rejected", "Below threshold confidence"

@app.post("/validate-id", response_model=IDValidationResponse)
async def validate_id(request: IDValidationRequest):
    """Main endpoint to validate college ID card"""
    try:
        logger.info(f"Processing validation request for user: {request.user_id}")
        
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Get image classification score
        image_score = image_classifier.classify(image)
        logger.info(f"Image classification score: {image_score}")
        
        # Get OCR validation
        ocr_result = ocr_validator.validate(image)
        ocr_score = ocr_result['confidence']
        ocr_confidence = ocr_result.get('ocr_confidence', 0.0)

        # Print OCR extracted text for debugging
        logger.info("\n--- OCR Extracted Text ---\n" + ocr_result.get('extracted_text', ''))
        logger.info(f"OCR validation score: {ocr_score}, OCR confidence: {ocr_confidence}")
        
        # Template matching (simplified)
        template_score = 0.8 if ocr_result['has_required_fields'] else 0.2
        
        # Calculate final score
        final_score = calculate_final_score(image_score, ocr_score, template_score)
        
        # Determine label and status
        threshold = config["validation_threshold"]
        label, status, reason = determine_label_and_status(final_score, ocr_confidence, threshold)
        
        # Additional reason details
        if not ocr_result['has_required_fields']:
            reason += " - Missing required fields"
        if not ocr_result['college_approved']:
            reason += " - College not in approved list"
        
        response = IDValidationResponse(
            user_id=request.user_id,
            validation_score=round(final_score, 2),
            label=label,
            status=status,
            reason=reason,
            threshold=threshold
        )
        
        logger.info(f"Validation result: {response.dict()}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing validation request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/version")
async def get_version():
    """Get model version information"""
    return {
        "version": "1.0.0",
        "model_version": "college-id-v1",
        "components": {
            "image_classifier": "ResNet18-based",
            "ocr_validator": "Tesseract + Custom rules",
            "template_matcher": "Layout analysis"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
