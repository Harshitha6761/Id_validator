# College ID Validator

An AI-based microservice that analyzes uploaded college ID card images to determine if they are genuine or fake.

## Features

- **Image-level manipulation detection** using ResNet18-based classifier
- **OCR consistency check** with Tesseract for text extraction and validation
- **Template matching** against known college layouts
- **Configurable validation rules** with approved college lists
- **RESTful API** built with FastAPI
- **Fully offline** - no external API dependencies
- **Docker support** for easy deployment

## API Endpoints

### POST /validate-id
Analyzes an uploaded ID card image.

**Request:**
```json
{
    "user_id": "stu_2290",
    "image_base64": "<base64_encoded_image>"
}
```

**Response:**
```json
{
    "user_id": "stu_2290",
    "validation_score": 0.85,
    "label": "genuine",
    "status": "approved",
    "reason": "High confidence genuine ID",
    "threshold": 0.70
}
```

### GET /health
Returns service health status.

### GET /version
Returns model version information.

## Decision Rules

| Condition | Label | Status | Action |
|-----------|-------|--------|---------|
| Score > 0.85 | genuine | approved | Automatic approval |
| Score 0.6-0.85 OR low OCR confidence | suspicious | manual_review | Human review required |
| Score < 0.6 OR OCR fails | fake | rejected | Automatic rejection |

## Installation & Usage

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

3. Create config.json file with the provided configuration

4. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t college-id-validator .
```

2. Run the container:
```bash
docker run -p 8000:8000 college-id-validator
```

## Testing

Run the test suite:
```bash
python -m pytest test_main.py -v
```

## Configuration

Edit `config.json` to customize:

- `validation_threshold`: Minimum score for approval (default: 0.7)
- `ocr_min_fields`: Required fields for validation
- `approved_colleges`: List of recognized institutions

## Model Components

1. **Image Classifier**: ResNet18-based neural network for detecting manipulated images
2. **OCR Validator**: Tesseract-based text extraction with custom validation rules
3. **Template Matcher**: Layout analysis for college ID template verification

## File Structure

```
├── main.py                 # FastAPI application
├── image_classifier.py     # Image manipulation detection
├── ocr_validator.py       # OCR and text validation
├── config.json            # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── test_main.py          # Test suite
└── README.md             # Documentation
```

## Test Cases Coverage

- ✅ Clear college ID → genuine, approved
- ✅ Fake template → fake, rejected  
- ✅ Cropped/screenshot → suspicious, manual_review
- ✅ Poor OCR quality → suspicious, manual_review
- ✅ Non-ID image → fake, rejected

## Limitations

- Uses simulated training data (no real student information)
- OCR accuracy depends on image quality
- Template matching is simplified for demonstration
- Model weights are initialized (would need real training data for production)

## Security & Privacy

- No external API calls - fully offline operation
- No storage of uploaded images
- No real student data used in development
- Base64 encoding for secure image transmission 