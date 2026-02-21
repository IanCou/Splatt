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

def test_video_endpoint(video_path):
    url = "http://localhost:8000/process_video"
    files = {'file': open(video_path, 'rb')}
    data = {'prompt': 'Accurately find specific textual details from this video.'}
    
    print(f"Sending {video_path} to {url}...")
    print("Note: Video processing may take a minute or two.")
    try:
        # Long timeout for video processing
        response = requests.post(url, files=files, data=data, timeout=600)
        response.raise_for_status()
        result = response.json()
        print("\nSuccess!")
        print(f"Filename: {result['filename']}")
        print(f"Video Analysis: {result['analysis']}")
    except Exception as e:
        print(f"\nError: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("Usage: python test_api.py <path_to_image_or_video>")
        sys.exit(1)
        
    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp']:
            test_vlm_endpoint(path)
        elif ext in ['.mp4', '.mpeg', '.mov', '.avi', '.flv', '.mpg', '.webm', '.wmv']:
            test_video_endpoint(path)
        else:
            print(f"Unsupported file extension: {ext}")
    else:
        print(f"File not found at {path}")
