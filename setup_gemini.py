#!/usr/bin/env python3
"""
Setup script to configure Gemini API key for Prompt Manager.
This sets the environment variable permanently on Windows.
"""

import os
import subprocess
import sys

def setup_gemini_api_key():
    """Setup Gemini API key as an environment variable."""
    print("=" * 60)
    print("Prompt Manager - Gemini API Setup")
    print("=" * 60)
    print()
    
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ API key cannot be empty!")
        sys.exit(1)
    
    # Set environment variable temporarily
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Set environment variable permanently on Windows
    try:
        subprocess.run(
            f'setx GEMINI_API_KEY "{api_key}"',
            shell=True,
            check=True,
            capture_output=True
        )
        print()
        print("✅ Success! Gemini API key has been configured.")
        print("   The API key is now stored as an environment variable.")
        print()
        print("📝 Note: You may need to restart your terminal or IDE")
        print("   for the changes to take effect.")
        print()
        print("🚀 You can now use the AI features in Prompt Manager!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting environment variable: {e}")
        print()
        print("Alternative: Set it manually in Windows:")
        print("1. Press Win + R")
        print("2. Type: sysdm.cpl")
        print("3. Go to Advanced > Environment Variables")
        print("4. Click 'New' and add:")
        print(f"   Variable name: GEMINI_API_KEY")
        print(f"   Variable value: {api_key}")
        sys.exit(1)

if __name__ == "__main__":
    setup_gemini_api_key()
