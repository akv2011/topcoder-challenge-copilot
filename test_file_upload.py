#!/usr/bin/env python3
"""
Test file upload functionality specifically
"""

import requests
import json
from PIL import Image
import io

def test_file_upload_detailed():
    """Test file upload with detailed logging"""
    backend_url = "http://localhost:8000"
    
    print("🧪 TESTING FILE UPLOAD FUNCTIONALITY")
    print("=" * 50)
    
    # First, create a challenge to upload to
    print("1. Creating a test challenge...")
    create_response = requests.post(
        f"{backend_url}/challenges",
        json={"goal": "Test challenge for file upload"}
    )
    
    if create_response.status_code != 200:
        print(f"❌ Failed to create challenge: {create_response.status_code}")
        print(f"Response: {create_response.text}")
        return False
    
    challenge_data = create_response.json()
    challenge_id = challenge_data["id"]
    print(f"✅ Challenge created: {challenge_id}")
    
    # Wait a moment for challenge creation to complete
    import time
    time.sleep(1)
    
    # Create a simple test image
    print("2. Creating test image...")
    img = Image.new('RGB', (100, 100), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    print("3. Uploading test image...")
    files = {"file": ("test_image.jpg", img_buffer.getvalue(), "image/jpeg")}
    data = {"challenge_id": challenge_id}
    
    print(f"   Uploading to: {backend_url}/upload")
    print(f"   Challenge ID: {challenge_id}")
    print(f"   File size: {len(img_buffer.getvalue())} bytes")
    
    upload_response = requests.post(
        f"{backend_url}/upload",
        files=files,
        data=data
    )
    
    print(f"4. Upload response:")
    print(f"   Status code: {upload_response.status_code}")
    print(f"   Headers: {dict(upload_response.headers)}")
    
    if upload_response.status_code == 200:
        response_data = upload_response.json()
        print(f"✅ Upload successful!")
        print(f"   Response: {json.dumps(response_data, indent=2)}")
        
        # Check if the challenge dialogue was updated
        print("5. Checking challenge dialogue update...")
        challenge_response = requests.get(f"{backend_url}/challenges/{challenge_id}")
        if challenge_response.status_code == 200:
            updated_challenge = challenge_response.json()
            dialogue_length = len(updated_challenge["dialogue_history"])
            print(f"   Dialogue length: {dialogue_length}")
            
            if dialogue_length > 2:  # Should have system, user, and new system message
                last_message = updated_challenge["dialogue_history"][-1]
                print(f"   Last message role: {last_message['role']}")
                print(f"   Last message preview: {last_message['content'][:100]}...")
                
                if "📎" in last_message["content"]:
                    print("✅ File analysis message found in dialogue!")
                    return True
                else:
                    print("❌ File analysis message not found in dialogue")
                    return False
            else:
                print("❌ Dialogue not updated after upload")
                return False
        else:
            print(f"❌ Failed to get updated challenge: {challenge_response.status_code}")
            return False
    else:
        print(f"❌ Upload failed!")
        print(f"   Error: {upload_response.text}")
        return False

if __name__ == "__main__":
    success = test_file_upload_detailed()
    if success:
        print("\n🎉 FILE UPLOAD TEST PASSED!")
    else:
        print("\n❌ FILE UPLOAD TEST FAILED!")
