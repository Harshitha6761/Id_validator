import requests
import base64
import argparse
import json

def validate_image(image_path: str, user_id: str = "manual_test_user"):
    """
    Reads an image file, encodes it to base64, and sends it to the
    validation API endpoint.
    """
    try:
        # Read image and encode to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the payload
        payload = {
            "user_id": user_id,
            "image_base64": image_base64
        }

        # API endpoint URL
        api_url = "http://localhost:8000/validate-id"

        print(f"Sending request for image: {image_path}")
        
        # Send the request
        response = requests.post(api_url, json=payload, timeout=30)

        # Print the results
        print("\n--- API Response ---")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {image_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a single ID card image.")
    parser.add_argument("image_path", type=str, help="The path to the image file to validate.")
    parser.add_argument("--user-id", type=str, default="manual_test_user_01", help="A user ID for the request.")
    
    args = parser.parse_args()
    
    validate_image(args.image_path, args.user_id) 