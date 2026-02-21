import requests
import sys
import os

def test_vlm_endpoint(image_path):
    url = "http://localhost:8000/process"
    files = {'file': open(image_path, 'rb')}
    data = {'prompt': 'What is in this image?'}
    
    print(f"Sending {image_path} to {url}...")
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        print("\nSuccess!")
        print(f"Filename: {result['filename']}")
        print(f"VLM Analysis: {result['analysis']}")
    except Exception as e:
        print(f"\nError: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Look for the generated image
        img_path = "test_image_for_vlm.png"
        
    if os.path.exists(img_path):
        test_vlm_endpoint(img_path)
    else:
        print(f"Image not found at {img_path}")
