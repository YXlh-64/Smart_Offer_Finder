#!/usr/bin/env python3
"""
Utility script to reload the chain on the running server
"""

import requests
import sys

def reload_chain(server_url="http://localhost:8000"):
    """Reload the chain on the running server"""
    
    try:
        print("🔄 Reloading chain on server...")
        response = requests.post(f"{server_url}/reload", timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Chain reloaded successfully!")
        print("\nCurrent Settings:")
        for key, value in result.get("settings", {}).items():
            print(f"  - {key}: {value}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server.")
        print(f"   Make sure the server is running at {server_url}")
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error: {e}")
        print(f"   Response: {e.response.text}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = reload_chain(server_url)
    sys.exit(0 if success else 1)
