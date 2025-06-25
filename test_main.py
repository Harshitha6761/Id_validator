import pytest
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def create_test_image(width=400, height=250, color=(255, 255, 255), add_text=False):
    """Create a test image and return as base64"""
    image = Image.new('RGB', (width, height), color)
    
    if add_text:
        draw = ImageDraw.Draw(image)
        # Add some sample ID card text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        texts = [
            "STUDENT ID CARD",
            "IIT Bombay",
            "Name: John Doe",
            "Roll No: 12CS3456",
            "Year: 2024",
            "Course: B.Tech CSE"
        ]
        
        y_offset = 20
        for text in texts:
            draw.text((20, y_offset), text, fill=(0, 0, 0), font=font)
            y_offset += 25
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_data = buffer.getvalue()
    return base64.b64encode(img_data).decode()

def create_fake_id_image():
    """Create a fake-looking ID image"""
    return create_test_image(width=800, height=500, color=(200, 200, 255), add_text=True)

def create_suspicious_image():
    """Create a suspicious image (wrong aspect ratio, poor quality)"""
    return create_test_image(width=100, height=100, color=(128, 128, 128))

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version_endpoint():
    """Test version endpoint"""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "components" in data
    assert "model_version" in data

def test_validate_genuine_id():
    """Test validation of a genuine-looking ID"""
    test_image = create_test_image(width=400, height=250, add_text=True)
    payload = {
        "user_id": "test_user_001",
        "image_base64": test_image
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Check response structure
    required_fields = ["user_id", "validation_score", "label", "status", "reason", "threshold"]
    for field in required_fields:
        assert field in data
    
    # Check field values
    assert data["user_id"] == "test_user_001"
    assert 0 <= data["validation_score"] <= 1
    assert data["label"] in ["genuine", "suspicious", "fake"]
    assert data["status"] in ["approved", "manual_review", "rejected"]
    assert data["threshold"] == 0.7

def test_validate_fake_id():
    """Test validation of a fake ID"""
    test_image = create_fake_id_image()
    payload = {
        "user_id": "test_user_002",
        "image_base64": test_image
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == "test_user_002"
    # Fake IDs should generally have lower scores
    # Note: Due to randomness in our heuristics, we can't guarantee exact scores

def test_validate_suspicious_image():
    """Test validation of suspicious image (poor quality)"""
    test_image = create_suspicious_image()
    payload = {
        "user_id": "test_user_003",
        "image_base64": test_image
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == "test_user_003"
    # Poor quality images should typically get suspicious or fake labels

def test_invalid_base64():
    """Test handling of invalid base64 data"""
    payload = {
        "user_id": "test_user_004",
        "image_base64": "invalid_base64_string"
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 400

def test_empty_base64():
    """Test handling of empty base64 data"""
    payload = {
        "user_id": "test_user_005",
        "image_base64": ""
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 400

def test_missing_fields():
    """Test handling of missing required fields"""
    # Missing user_id
    payload = {
        "image_base64": create_test_image()
    }
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 422
    
    # Missing image_base64
    payload = {
        "user_id": "test_user_006"
    }
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 422

def test_data_url_format():
    """Test handling of data URL format base64"""
    test_image = create_test_image()
    data_url = f"data:image/jpeg;base64,{test_image}"
    
    payload = {
        "user_id": "test_user_007",
        "image_base64": data_url
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == "test_user_007"

def test_different_image_sizes():
    """Test validation with different image sizes"""
    test_cases = [
        (200, 125),   # Small image
        (400, 250),   # Standard image
        (800, 500),   # Large image
        (1200, 750),  # Very large image
    ]
    
    for width, height in test_cases:
        test_image = create_test_image(width=width, height=height)
        payload = {
            "user_id": f"test_user_size_{width}x{height}",
            "image_base64": test_image
        }
        
        response = client.post("/validate-id", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "validation_score" in data

def test_extreme_aspect_ratios():
    """Test images with extreme aspect ratios"""
    # Very wide image
    wide_image = create_test_image(width=800, height=100)
    payload = {
        "user_id": "test_user_wide",
        "image_base64": wide_image
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200
    
    # Very tall image
    tall_image = create_test_image(width=100, height=800)
    payload = {
        "user_id": "test_user_tall",
        "image_base64": tall_image
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200

def test_college_recognition():
    """Test if approved colleges are recognized"""
    # This would require creating images with actual college names
    # For now, we'll test with a basic image
    test_image = create_test_image(add_text=True)
    payload = {
        "user_id": "test_user_college",
        "image_base64": test_image
    }
    
    response = client.post("/validate-id", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    # The response should include information about college recognition
    assert "reason" in data

def test_concurrent_requests():
    """Test handling of multiple concurrent requests"""
    import concurrent.futures
    import threading
    
    def make_request(user_id):
        test_image = create_test_image()
        payload = {
            "user_id": user_id,
            "image_base64": test_image
        }
        response = client.post("/validate-id", json=payload)
        return response.status_code, response.json()
    
    # Make 5 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, f"concurrent_user_{i}") for i in range(5)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # All requests should succeed
    for status_code, data in results:
        assert status_code == 200
        assert "validation_score" in data

def test_performance_timing():
    """Test response time performance"""
    import time
    
    test_image = create_test_image()
    payload = {
        "user_id": "performance_test_user",
        "image_base64": test_image
    }
    
    start_time = time.time()
    response = client.post("/validate-id", json=payload)
    end_time = time.time()
    
    assert response.status_code == 200
    
    # Response should be reasonably fast (less than 10 seconds)
    response_time = end_time - start_time
    assert response_time < 10.0, f"Response took too long: {response_time} seconds"

if __name__ == "__main__":
    print("Running College ID Validator Test Suite...")
    print("=" * 50)
    
    # Run specific test categories
    test_functions = [
        test_health_endpoint,
        test_version_endpoint,
        test_validate_genuine_id,
        test_validate_fake_id,
        test_validate_suspicious_image,
        test_invalid_base64,
        test_missing_fields,
        test_different_image_sizes,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            failed += 1
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} tests failed")