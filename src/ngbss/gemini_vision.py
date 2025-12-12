"""
Gemini Flash Vision API with Key Rotation
Handles rate limiting by rotating through multiple API keys
"""

import google.generativeai as genai
from PIL import Image
import time
import random
from typing import Optional

# Multiple API keys for rotation (to avoid rate limits)
GOOGLE_KEYS = [
    "AIzaSyCV3SgE-HF0dQHUXyFM3iqDfC7kpH7PWp4", "AIzaSyD5jnbXhUR7CxbTE8PM_FvkJE0rO1Wf88o",
    "AIzaSyA_dYhpFkS5UubBPFvqx1muj8Rdr0P0Pp4", "AIzaSyB4inzukUQytYA6n6U8zBGEI1lhmHXxLCU",
    "AIzaSyCcsI4Tfr1jWObA50TtJFVHadgePNBgQ_Y", "AIzaSyBOrB59jCkC8rteXcMQpFZCtg60kgb8xGQ",
    "AIzaSyBlV31WBKMxaUQnsJObKTgkGD008R6tMk8", "AIzaSyAb9NUAaGhRs3Pg5bqaJ1IPf9mCxXn7Udg",
    "AIzaSyAOEEYTxQWZnzt3oQJ7pj4V3qkAJWpGdFQ", "AIzaSyBEakNS2YqXjRaXUUReqGczlJzhYQ1sD_U",
    "AIzaSyCWoRhbc700YAVcFhx7u-i84aujieP40VI", "AIzaSyDKDb12g5B9gb3wBs9wGB2y1ZSbgnH4doE",
    "AIzaSyBoBXF4tl5Q5B7JmNpNCUaYANiq3M2MvZY", "AIzaSyBPxH9lCMuFsoW0LT1dkKl7TDhu2bqpadE",
    "AIzaSyCtu_lC0FNmCgGV5sV9okAbO3740YB8JcY", "AIzaSyAbHTWRLEXQLjRCFmRlT897HDWhWVYHGrc",
    "AIzaSyA3Nmk_XtRLy9lciPC1SgF42wlbnNGDrw0", "AIzaSyBAXNRjcmG6LtioT99_knzo52CWMa2m5tY",
    "AIzaSyC6INNh6fQAkbnsCLocfZxkO2lqTkKJW1M", "AIzaSyCpjN9f4i7kp1WKvpeCUpsDjf5HyHjuciE",
    "AIzaSyArxW5qjaevGdBZFzSKmCyPSFZwAxq1498", "AIzaSyBLuiHwA5tpYkuRwgWlUObvXtJNx6Vwfd8",
    "AIzaSyBq_1XqJUYp1FPWf2MvLvkT07o2dtC3MxY", "AIzaSyAiE8gpdYusufodg3mIyO7i_eiw5kpfak8",
    "AIzaSyAAouUGJPboUOllFq0ZkQ8sOBK1Kl-1lr0", "AIzaSyCrUWr1OvdQh1ETUPloc6YbeIVH8giEBBc",
    "AIzaSyDAOOUFBoczaN3TGmMUOPl7P03NEL1DHG8", "AIzaSyDfMsTrche6Mrjw7sTYD0r_G-00kZoWYXU",
    "AIzaSyA129jTBwc1KLsACynJ-Qou5PYi3OhfsTI", "AIzaSyA2oA7Z_lZQ7suRBMEOJBIUA6nEhpqjOlE",
    "AIzaSyBPdxQpQKgl_xanRdOAzM6_-X8FQODpCr4", "AIzaSyBpYRjItjkvXJ7dyNwNa13-tohUhlbBn84",
    "AIzaSyBOrB59jCkC8rteXcMQpFZCtg60kgb8xGQ", "AIzaSyAO-UvPAIUqSpYOiUuD61Fd3yIRqkhryfY",
    "AIzaSyCcub1T9Vd2n_iP3cm-1FbJ6NvJASQNID0", "AIzaSyCTXvjKUM4FctW9clvqzy1LjL-ZrmDKMRc",
    "AIzaSyCYPOERL_ZrU57dQIuyL_aGtac-C3OqZno",
]


class GeminiVision:
    """Gemini Flash Vision API with automatic key rotation."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.keys = GOOGLE_KEYS.copy()
        random.shuffle(self.keys)  # Randomize to distribute load
        self.current_key_index = 0
        self.failed_keys = set()
        self._configure_current_key()
    
    def _configure_current_key(self):
        """Configure genai with current API key."""
        key = self.keys[self.current_key_index]
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.model_name)
        print(f"🔑 Using API key #{self.current_key_index + 1}")
    
    def _rotate_key(self):
        """Switch to next available API key."""
        self.failed_keys.add(self.current_key_index)
        
        # Find next working key
        for _ in range(len(self.keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            if self.current_key_index not in self.failed_keys:
                self._configure_current_key()
                return True
        
        print("❌ All API keys exhausted!")
        return False
    
    def extract_from_image(self, image: Image.Image, procedure_name: str, 
                           step_number: int, max_retries: int = 3) -> Optional[dict]:
        """
        Extract structured information from an NGBSS screenshot.
        
        Args:
            image: PIL Image of the PDF page
            procedure_name: Name of the procedure (e.g., "Recharge par Bon de Commande")
            step_number: Step number in the procedure
            max_retries: Number of retries with key rotation
        
        Returns:
            dict with extracted information or None if failed
        """
        
        prompt = f"""Analyze this screenshot from the NGBSS guide for Algérie Télécom.

**Procedure:** {procedure_name}
**Step Number:** {step_number}

Extract the following information in French:

1. **ACTION**: What action must the user perform? (e.g., "Cliquer sur...", "Saisir...", "Sélectionner...")
2. **VISUAL_LOCATION**: Where is the element located on screen? (e.g., "en haut à droite", "dans le menu latéral gauche")
3. **FIELDS**: List any form fields visible with their labels
4. **BUTTONS**: List any buttons visible with their text
5. **IMPORTANT_VALUES**: Any specific values shown (amounts, account numbers, etc.)
6. **WARNINGS**: Any warnings, required fields (marked with *), or important notes
7. **NAVIGATION**: Menu path or tabs to reach this screen (if visible)

Respond in this exact JSON format:
{{
    "action": "Description de l'action à effectuer",
    "visual_location": "Position de l'élément sur l'écran",
    "fields": ["Champ 1", "Champ 2"],
    "buttons": ["Bouton 1", "Bouton 2"],
    "important_values": ["Valeur importante 1"],
    "warnings": ["Avertissement ou note importante"],
    "navigation": "Menu > Sous-menu > Onglet",
    "summary": "Résumé complet de cette étape en une phrase"
}}

If the image is not clear or doesn't show NGBSS content, return:
{{"error": "Description du problème"}}
"""
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content([prompt, image])
                
                # Parse JSON from response
                text = response.text.strip()
                
                # Extract JSON from markdown code blocks if present
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                import json
                result = json.loads(text)
                
                # Add metadata
                result["procedure_name"] = procedure_name
                result["step_number"] = step_number
                
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Rate limit or quota exceeded - rotate key
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    print(f"⚠️ Rate limit hit, rotating key...")
                    if not self._rotate_key():
                        return None
                    time.sleep(1)
                    
                # Other errors
                else:
                    print(f"❌ Error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
        
        return None
    
    def test_connection(self) -> bool:
        """Test if Gemini API is accessible. Rotates through keys until one works."""
        max_attempts = len(self.keys)
        
        for attempt in range(max_attempts):
            try:
                response = self.model.generate_content("Say 'OK' if you can read this.")
                if response and response.text:
                    print(f"✅ Key #{self.current_key_index + 1} is working!")
                    return True
            except Exception as e:
                error_msg = str(e).lower()
                
                # Rate limit / quota - try next key
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    print(f"⚠️ Key #{self.current_key_index + 1} quota exceeded, trying next...")
                    if not self._rotate_key():
                        print("❌ All keys exhausted!")
                        return False
                    time.sleep(0.5)
                else:
                    print(f"❌ Connection test failed: {e}")
                    return False
        
        return False


# Quick test
if __name__ == "__main__":
    vision = GeminiVision()
    if vision.test_connection():
        print("✅ Gemini Flash is working!")
    else:
        print("❌ Failed to connect to Gemini")
