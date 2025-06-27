# ocr_validator.py
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
from typing import Dict, List, Any
import difflib
import logging

class OCRValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.approved_colleges = config.get("approved_colleges", [])
        self.required_fields = config.get("ocr_min_fields", ["name", "roll_number", "college"])
    
    def validate(self, image: Image.Image) -> Dict[str, Any]:
        """Validate ID card using OCR and text analysis"""
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Extract text using OCR
            extracted_text = self._extract_text(processed_image)
            
            # Analyze extracted text
            analysis = self._analyze_text(extracted_text)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(analysis, extracted_text)
            
            return {
                'confidence': confidence_score,
                'ocr_confidence': analysis['ocr_quality'],
                'extracted_text': extracted_text,
                'has_required_fields': analysis['has_required_fields'],
                'college_approved': analysis['college_approved'],
                'detected_fields': analysis['detected_fields']
            }
            
        except Exception as e:
            print(f"Error in OCR validation: {e}")
            return {
                'confidence': 0.1,
                'ocr_confidence': 0.0,
                'extracted_text': "",
                'has_required_fields': False,
                'college_approved': False,
                'detected_fields': {}
            }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert to OpenCV format for additional processing
        cv_image = cv2.cvtColor(np.array(blurred), cv2.COLOR_GRAY2BGR)
        
        # Apply adaptive thresholding
        gray_cv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL
        return Image.fromarray(thresh)
    
    def _extract_text(self, image: Image.Image) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            # Configure Tesseract for better ID card recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/() '
            
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ""
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze extracted text for ID card validation"""
        analysis = {
            'has_required_fields': False,
            'college_approved': False,
            'detected_fields': {},
            'ocr_quality': 0.0
        }
        
        if not text:
            return analysis
        
        text_lower = text.lower()
        lines = text.split('\n')
        
        # Calculate OCR quality based on text characteristics
        analysis['ocr_quality'] = self._calculate_ocr_quality(text)
        
        # Detect various fields
        detected_fields = {}
        
        # --- Robust College Name Matching ---
        def normalize(s):
            return ''.join(e for e in s.lower() if e.isalnum())
        norm_text = normalize(text)
        best_match = None
        best_score = 0.0
        for college in self.approved_colleges:
            norm_college = normalize(college)
            seq = difflib.SequenceMatcher(None, norm_college, norm_text)
            score = seq.find_longest_match(0, len(norm_college), 0, len(norm_text)).size / max(1, len(norm_college))
            if score > best_score:
                best_score = score
                best_match = college
        if best_score > 0.7:  # Allow for some OCR error
            detected_fields['college'] = best_match
            analysis['college_approved'] = True
        
        # --- Robust Roll Number Patterns ---
        roll_patterns = [
            r'roll\s*(no|number|num|#)?[:\s-]*([A-Z0-9]+)',  # Roll No: ABC123
            r'reg\s*(no|number|num|#)?[:\s-]*([A-Z0-9]+)',   # Reg No: XYZ456
            r'\b\d{2}[A-Z]{2}\d{4}\b',  # 12AB3456
            r'\b[A-Z]{2}\d{6}\b',       # AB123456
            r'\b\d{8,12}\b',            # 12345678
        ]
        for pattern in roll_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    detected_fields['roll_number'] = matches[0][-1]
                else:
                    detected_fields['roll_number'] = matches[0]
                break
        
        # --- Robust Name Patterns ---
        name_patterns = [
            r'name[:\s-]+([A-Za-z\s]+)',
            r'student[:\s-]+([A-Za-z\s]+)',
            r'bearer[:\s-]+([A-Za-z\s]+)',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                name = matches[0].strip()
                if 2 < len(name) < 50:
                    detected_fields['name'] = name
                break
        
        # Year/Course detection (unchanged)
        if re.search(r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s*(year|semester)', text_lower):
            detected_fields['year'] = True
        if re.search(r'\b(engineering|btech|mtech|bsc|msc|mba|phd)\b', text_lower):
            detected_fields['course'] = True
        
        analysis['detected_fields'] = detected_fields
        
        # Check if required fields are present
        field_count = 0
        for field in self.required_fields:
            if field in detected_fields or (field == 'college' and analysis['college_approved']):
                field_count += 1
        analysis['has_required_fields'] = field_count >= len(self.required_fields) - 1
        
        # --- Logging for debugging ---
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(f"OCR Extracted Text: {text}")
        logger.info(f"Detected Fields: {detected_fields}")
        logger.info(f"College Match Score: {best_score:.2f} ({best_match})")
        
        return analysis
    
    def _calculate_ocr_quality(self, text: str) -> float:
        """Calculate OCR quality score based on text characteristics"""
        if not text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check text length (too short or too long might indicate issues)
        if 20 <= len(text) <= 500:
            score += 0.2
        
        # Check for reasonable word count
        words = text.split()
        if 5 <= len(words) <= 100:
            score += 0.1
        
        # Check for mixed case (indicates proper text recognition)
        if any(c.isupper() for c in text) and any(c.islower() for c in text):
            score += 0.1
        
        # Check for numbers (ID cards typically have numbers)
        if any(c.isdigit() for c in text):
            score += 0.1
        
        # Penalize excessive special characters (might indicate OCR errors)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c != ' ') / len(text)
        if special_char_ratio > 0.3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, analysis: Dict[str, Any], text: str) -> float:
        """Calculate overall OCR validation confidence"""
        confidence = 0.0
        
        # Base confidence from OCR quality
        confidence += analysis['ocr_quality'] * 0.3
        
        # Bonus for required fields
        if analysis['has_required_fields']:
            confidence += 0.4
        
        # Bonus for approved college
        if analysis['college_approved']:
            confidence += 0.2
        
        # Bonus for number of detected fields
        field_bonus = min(len(analysis['detected_fields']) * 0.1, 0.3)
        confidence += field_bonus
        
        return max(0.0, min(1.0, confidence))

