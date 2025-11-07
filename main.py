import os
import warnings
import time
import random
import io
import json
import sys
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock 

# NEW IMPORT for serving frontend
from flask import Flask, jsonify, request, send_file, Response, render_template 
from flask_cors import CORS 
from dotenv import load_dotenv

# Web Scraping Libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Gemini API Libraries
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Configuration & Initialization ---

# Suppress warnings for insecure requests
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

# Global Flask App Placeholder
app = Flask(__name__)

# --- Configuration Class for Centralized Settings ---
class Config:
    """Centralized configuration and state management."""
    
    # VTU URLs
    DEFAULT_INDEX_URL = 'https://results.vtu.ac.in/JJEcbcs25/index.php'
    DEFAULT_RESULT_URL = 'https://results.vtu.ac.in/JJEcbcs25/resultpage.php'
    
    # Gemini Settings
    MODEL_NAME = 'gemini-2.5-flash' # Updated from pro back to flash for SPEED
    API_KEYS: List[str] = []
    
    # Concurrency Settings
    MAX_SCRAPER_WORKERS = 20 # Increased from 10 to 20
    MAX_RETRY_ATTEMPTS = 5
    SCRAPER_BASE_DELAY = 1.5 # Increased from 0.5 to 1.5 seconds
    
    # Cache/Storage Settings
    # In a real advanced app, TEMP_EXCEL_STORAGE would be replaced by Redis/S3.
    TEMP_EXCEL_STORAGE: Dict[str, io.BytesIO] = {} 
    
    @classmethod
    def load_keys(cls):
        """Load standard and fallback API keys."""
        if os.getenv("GEMINI_API_KEY"):
            cls.API_KEYS.append(os.getenv("GEMINI_API_KEY"))
        for i in range(1, 10): 
            key_name = f"GEMINI_API_KEY{i}"
            key_value = os.getenv(key_name)
            if key_value:
                cls.API_KEYS.append(key_value)
        return len(cls.API_KEYS)

# Load all keys before application start
Config.load_keys()


# --- Thread-Safe Key Rotation and Rate Limiting (IMPROVED) ---
class KeyRotator:
    """
    Manages API key rotation, tracks last usage, and applies an exponential 
    backoff penalty for rate-limited keys.
    """
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.num_keys = len(api_keys)
        self.current_index = 0
        self.lock = Lock()
        self.last_used: List[float] = [0.0] * self.num_keys
        self.penalty_time: List[float] = [0.0] * self.num_keys
        self.base_cool_down = 1.0 # Reduced from 2.0 

    def penalize_key(self, key_index: int):
        """Applies an exponential backoff penalty for rate limiting."""
        with self.lock:
            current_penalty = self.penalty_time[key_index]
            new_penalty = min(current_penalty * 2.0 or 5.0, 60.0) 
            self.penalty_time[key_index] = new_penalty
            app.logger.warning(f"Key #{key_index + 1} penalized. Next available in {new_penalty:.1f}s.")
        
    def get_next_client(self) -> Tuple[genai.Client, str, int] | Tuple[None, None, None]:
        """
        Rotates to the next key, enforces a cool-down/penalty, and returns a new client.
        """
        with self.lock:
            if not self.api_keys:
                return None, None, None
            
            for _ in range(self.num_keys):
                key_index = self.current_index
                api_key = self.api_keys[key_index]
                key_alias = f"Key #{key_index + 1}"
                
                required_cool_down = max(self.base_cool_down, self.penalty_time[key_index])
                time_since_last_use = time.time() - self.last_used[key_index]
                
                if time_since_last_use < required_cool_down:
                    self.current_index = (self.current_index + 1) % self.num_keys
                    continue

                self.penalty_time[key_index] = 0.0 
                self.current_index = (self.current_index + 1) % self.num_keys
                
                try:
                    client = genai.Client(api_key=api_key)
                    self.last_used[key_index] = time.time() 
                    return client, key_alias, key_index
                except Exception as e:
                    app.logger.error(f"Client Init Error ({key_alias}): {e}")
            
            return None, None, None


# --- Core CAPTCHA Solver Class (Decoupled) ---

class CaptchaSolver:
    """Manages Gemini-based CAPTCHA solving with Key Rotation."""
    
    def __init__(self, key_rotator: KeyRotator):
        self.key_rotator = key_rotator

    def solve(self, image_content: bytes) -> Optional[str]:
        """Core CAPTCHA solving logic using a rotating, rate-limited key."""
        
        MAX_ATTEMPTS = self.key_rotator.num_keys * 3 

        for attempt in range(MAX_ATTEMPTS):
            client, key_alias, key_index = self.key_rotator.get_next_client()
            
            if client is None:
                app.logger.warning(f"All keys penalized or cooling down. Waiting for {Config.SCRAPER_BASE_DELAY * 2}s.")
                time.sleep(Config.SCRAPER_BASE_DELAY * 2) 
                continue 

            try:
                # Define the Structured Output Schema
                captcha_schema = types.Schema(
                    type=types.Type.OBJECT,
                    properties={"captcha_code": types.Schema(type=types.Type.STRING)},
                    required=["captcha_code"]
                )

                prompt_text = (
                    "Analyze the CAPTCHA image. Extract ONLY the 6-character alphanumeric code (A-Z, 0-9) "
                    "in a JSON object that adheres to the provided schema."
                )
                
                contents = [
                    types.Part.from_text(text=prompt_text),
                    types.Part.from_bytes(data=image_content, mime_type='image/png')
                ]
                
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=captcha_schema,
                    temperature=0.0
                )

                response = client.models.generate_content(
                    model=Config.MODEL_NAME, contents=contents, config=config
                )

                json_data = json.loads(response.text)
                captcha_code = json_data.get("captcha_code", "").strip()
                
                if len(captcha_code) == 6 and captcha_code.isalnum():
                    app.logger.info(f"-> AI Solution ({key_alias}): '{captcha_code}'")
                    return captcha_code
                
            except APIError as e:
                if 'RESOURCE_EXHAUSTED' in str(e) or '429' in str(e):
                    self.key_rotator.penalize_key(key_index) 
                    continue 
                app.logger.error(f"-> GEMINI API Error ({key_alias}): {e}")
            except (json.JSONDecodeError, Exception) as e:
                app.logger.error(f"-> Unexpected error during AI solving ({key_alias}): {type(e).__name__} - {e}")
            
        app.logger.error("ALL API KEY attempts exhausted for the current CAPTCHA solving.")
        return None

# --- Core Scraper Logic Class (Decoupled) ---

class VTUScraper:
    """Encapsulates all web scraping logic."""
    
    def __init__(self, captcha_solver: CaptchaSolver):
        self.captcha_solver = captcha_solver

    def fetch_result(self, usn: str, index_url: str, result_url: str) -> Optional[dict]:
        """Fetch VTU result for a given USN with automatic retry."""
        
        session = requests.Session()
        retries = Retry(total=Config.MAX_RETRY_ATTEMPTS, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Referer': index_url 
        }
        
        for attempt in range(1, Config.MAX_RETRY_ATTEMPTS + 1):
            app.logger.info(f"[{usn} | Attempt {attempt}] Starting fetch...")
            
            try:
                # 1. Get Token and CAPTCHA link
                session.cookies.clear()
                r = session.get(index_url, headers=headers, verify=False, timeout=10)
                if r.status_code != 200:
                    time.sleep(random.uniform(1, 3)); continue
                
                soup = BeautifulSoup(r.text, 'html.parser')
                token_tag = soup.find('input', {'name': 'Token'})
                captcha_img = soup.find('img', alt='CAPTCHA') or soup.find('img', src=lambda s: s and 'captcha' in s.lower())

                if not token_tag or not captcha_img:
                    app.logger.error(f"[{usn}] Error: Missing Token or CAPTCHA image on index page. Aborting.")
                    return None
                
                token = token_tag['value']
                captcha_src = urljoin(index_url, captcha_img['src'])
                
                # 2. Fetch CAPTCHA image and solve
                captcha_r = session.get(captcha_src, headers=headers, verify=False, timeout=10)
                if captcha_r.status_code != 200:
                    time.sleep(random.uniform(1, 3)); continue
                
                captcha_code = self.captcha_solver.solve(captcha_r.content)
                if not captcha_code:
                    time.sleep(random.uniform(1, 3)); continue 
                
                # 3. Submit Result
                data = {'Token': token, 'lns': usn, 'captchacode': captcha_code}
                post_r = session.post(result_url, data=data, headers=headers, verify=False, timeout=10)
                
                if post_r.status_code != 200:
                    time.sleep(random.uniform(1, 3)); continue
                
                # 4. Check for success/failure
                text_lower = post_r.text.lower()
                if 'invalid captcha code' in text_lower or 'wrong captcha' in text_lower:
                    app.logger.warning(f"[{usn}] POST failed: Invalid CAPTCHA Code submitted.")
                    time.sleep(random.uniform(1, 3)); continue 
                
                if 'student name' not in text_lower and 'university seat number' not in text_lower:
                    return None # USN invalid or not found
                
                # 5. Robust Parsing (FIXED Index Error)
                result_soup = BeautifulSoup(post_r.text, 'html.parser')
                
                name = "Unknown"
                try:
                    # New, more robust parsing logic
                    # Find the <b> tag containing 'Student Name'
                    name_label_tag = result_soup.find('b', string=lambda t: t and 'Student Name' in t)
                    
                    if name_label_tag:
                        # The name is often in the next sibling <td> element after the parent <td>
                        # Structure can be: <td><b>Name</b></td> <td>:</td> <td>NAME</td>
                        
                        # Go up to the parent <td>
                        parent_td = name_label_tag.find_parent('td')
                        
                        # Find the next <td> sibling
                        next_td = parent_td.find_next_sibling('td')
                        
                        if next_td:
                            # Check if it's the colon <td>
                            if next_td.get_text(strip=True) == ':':
                                # If it's a colon, the name is in the *next* sibling after that
                                name_td = next_td.find_next_sibling('td')
                                if name_td:
                                    name = name_td.get_text(strip=True)
                            else:
                                # If not a colon, this <td> might be the name
                                name = next_td.get_text(strip=True)
                        
                        # Fallback: if the name is in the same <td> (like "Student Name : NAME")
                        if not name or name == "Unknown":
                             full_text = parent_td.get_text(strip=True)
                             if ':' in full_text:
                                parts = full_text.split(':', 1)
                                if len(parts) == 2 and parts[1].strip():
                                    name = parts[1].strip()

                    # If the new logic fails, try the old logic as a final fallback
                    if not name or name == "Unknown":
                        for td in result_soup.find_all('td'):
                            td_text = td.get_text(strip=True)
                            if 'Student Name' in td_text:
                                parts = td_text.split(':', 1)
                                if len(parts) == 2 and parts[1].strip():
                                    name = parts[1].strip()
                                    break # Found it
                except Exception as parse_e:
                    app.logger.error(f"[{usn}] Error during name parsing: {parse_e}")
                    name = "Unknown" # Ensure it's set to Unknown on parse error
                
                semester = "Unknown"
                for div in result_soup.find_all('div', string=lambda t: t and 'Semester' in t):
                    sem_text = div.get_text(strip=True)
                    parts = sem_text.split(':', 1)
                    if len(parts) == 2:
                        semester = parts[1].strip()
                        break
                
                subjects: List[Dict[str, str]] = []
                table_body = result_soup.find('div', {'class': 'divTableBody'})
                if table_body:
                    rows = table_body.find_all('div', {'class': 'divTableRow'})
                    for row in rows[1:]:
                        cells = row.find_all('div', {'class': 'divTableCell'})
                        if len(cells) >= 7:
                            subjects.append({
                                'code': cells[0].get_text(strip=True),
                                'name': cells[1].get_text(strip=True),
                                'internal': cells[2].get_text(strip=True),
                                'external': cells[3].get_text(strip=True),
                                'total': cells[4].get_text(strip=True),
                                'result': cells[5].get_text(strip=True),
                                'announced': cells[6].get_text(strip=True)
                            })
                
                app.logger.info(f"[{usn}] ✓ Success: Result fetched for {name}.")
                time.sleep(random.uniform(Config.SCRAPER_BASE_DELAY, Config.SCRAPER_BASE_DELAY * 2)) 
                return {'usn': usn, 'name': name, 'semester': semester, 'subjects': subjects}

            except Exception as e:
                app.logger.error(f"[{usn}] Error processing on attempt {attempt}: {type(e).__name__} - {e}")
                time.sleep(random.uniform(1, 3))
                continue
        
        app.logger.error(f"[{usn}] ❌ Failed to retrieve result after {Config.MAX_RETRY_ATTEMPTS} attempts.")
        return None

    def get_bulk_results(self, usn_list: List[str], index_url: str, result_url: str, subject_code: str = '') -> Tuple[List[Dict], List[Dict]]:
        """Processes a list of USNs concurrently."""
        successful_results: List[Dict] = []
        failed_usns: List[Dict] = []
        
        app.logger.info(f"Starting concurrent fetch for {len(usn_list)} USNs with {Config.MAX_SCRAPER_WORKERS} workers.")

        def process_single_usn(usn: str):
            usn = usn.strip().upper()
            if not usn: return None
            
            raw_result = self.fetch_result(usn, index_url, result_url)
            
            if raw_result:
                if subject_code:
                    raw_result['subjects'] = [
                        sub for sub in raw_result['subjects'] 
                        if sub['code'].lower() == subject_code.lower()
                    ]
                return raw_result
            else:
                return {"usn": usn, "error": "Failed to retrieve result after multiple CAPTCHA attempts or USN is invalid/not found."}

        with ThreadPoolExecutor(max_workers=Config.MAX_SCRAPER_WORKERS) as executor:
            results_iterator = executor.map(process_single_usn, usn_list, timeout=None)
            
            for result in results_iterator:
                if result is None: continue
                if 'name' in result:
                    successful_results.append(result)
                elif 'error' in result:
                    failed_usns.append(result)
                    
        app.logger.info(f"Concurrent processing finished. Successful: {len(successful_results)}, Failed: {len(failed_usns)}.")
        return successful_results, failed_usns

# --- Utility Function (Excel) ---

def generate_bulk_excel_file(results_data: List[dict]) -> tuple[str, io.BytesIO]:
    """Converts results into a consolidated Excel file."""
    if not results_data:
        raise ValueError("No data provided for Excel generation.")

    consolidated_rows = []
    # ... (Excel consolidation logic is unchanged and robust) ...
    for result in results_data:
        usn = result['usn']
        name = result.get('name', 'N/A')
        semester = result.get('semester', 'N/A')
        student_base_data = {'USN': usn, 'Name': name, 'Semester': semester}
        
        for subject in result['subjects']:
            row = student_base_data.copy()
            row.update({
                'Subject Code': subject.get('code', ''),
                'Subject Name': subject.get('name', ''),
                'Internal Marks': subject.get('internal', ''),
                'External Marks': subject.get('external', ''),
                'Total Marks': subject.get('total', ''),
                'Result': subject.get('result', ''),
                'Announced Date': subject.get('announced', '')
            })
            consolidated_rows.append(row)

    consolidated_df = pd.DataFrame(consolidated_rows)
    COLUMNS_ORDER = [
        'USN', 'Name', 'Semester', 'Subject Code', 'Subject Name', 
        'Internal Marks', 'External Marks', 'Total Marks', 'Result', 'Announced Date'
    ]
    final_df = consolidated_df.reindex(columns=COLUMNS_ORDER, fill_value='')

    output = io.BytesIO()
    timestamp = int(time.time())
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='Consolidated Results', index=False)
    
    output.seek(0)
    base_usn = results_data[0].get('usn', 'Bulk')
    filename = f"VTU_Results_Consolidated_{base_usn}_{timestamp}.xlsx"
    
    return filename, output

# --- Flask App Setup and Routes ---

CORS(app) 
app.logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# Initialize components outside the Flask app definition (Dependency Injection)
KEY_ROTATOR = KeyRotator(Config.API_KEYS)
CAPTCHA_SOLVER = CaptchaSolver(KEY_ROTATOR)
VTU_SCRAPER = VTUScraper(CAPTCHA_SOLVER)

@app.route('/', methods=['GET'])
def index() -> Response:
    """Serves the main frontend HTML page."""
    return render_template('index.html', 
        default_index_url=Config.DEFAULT_INDEX_URL,
        default_result_url=Config.DEFAULT_RESULT_URL
    )

@app.route('/api/vtu/download/<filename>', methods=['GET'])
def download_excel(filename: str) -> Response:
    """Serves the temporarily stored Excel file and removes it from memory."""
    excel_stream = Config.TEMP_EXCEL_STORAGE.pop(filename, None)
    
    if excel_stream is None:
        return jsonify({"error": "File not found or link has expired. Please fetch the result again."}), 404
    
    app.logger.info(f"Serving and removing temporary file: {filename}")
    
    return send_file(
        excel_stream,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/vtu/results', methods=['POST'])
def get_bulk_vtu_results() -> Response:
    """API endpoint to fetch VTU results for a list of USNs."""
    
    try:
        request_data: Any = request.get_json(silent=True)
        if not request_data or not isinstance(request_data, dict):
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        usn_list_raw = request_data.get('usns')
        subject_code = str(request_data.get('subject_code', '')).strip()
        index_url = str(request_data.get('index_url', Config.DEFAULT_INDEX_URL)).strip()
        result_url = str(request_data.get('result_url', Config.DEFAULT_RESULT_URL)).strip()
        
        if not index_url.startswith('http') or not result_url.startswith('http'):
             return jsonify({"error": "Invalid 'index_url' or 'result_url'. Must be a complete URL starting with http/https."}), 400
        
        if not isinstance(usn_list_raw, list) or not usn_list_raw:
            return jsonify({"error": "Missing or invalid 'usns' list in the request body."}), 400
        
        usn_list = [str(u).strip() for u in usn_list_raw if str(u).strip()]
        
    except Exception as e:
        app.logger.error(f"Error parsing request body: {e}")
        return jsonify({"error": "Failed to parse request data."}), 400

    if not Config.API_KEYS:
         return jsonify({"error": "No Gemini API Keys are available. Cannot proceed with AI CAPTCHA solving."}), 500

    # Use the decoupled Scraper class instance
    successful_results, failed_usns = VTU_SCRAPER.get_bulk_results(usn_list, index_url, result_url, subject_code)
    
    download_url = "No Excel file generated (No successful results)."
    
    if successful_results:
        try:
            filename, excel_stream = generate_bulk_excel_file(successful_results)
            # Store file in the temporary in-memory storage defined in Config
            Config.TEMP_EXCEL_STORAGE[filename] = excel_stream 
            download_url = f"{request.url_root.rstrip('/')}/api/vtu/download/{filename}"
        except Exception as e:
            app.logger.error(f"Error generating Bulk Excel file: {e}")
            download_url = f"Error generating Excel file: {type(e).__name__}"

    response_data = {
        "status": "partial_success" if successful_results and failed_usns else ("success" if successful_results else "failure"),
        "total_requested": len(usn_list),
        "total_successful": len(successful_results),
        "total_failed": len(failed_usns),
        "download_url": download_url,
        "current_vtu_index_url": index_url,
        "current_vtu_result_url": result_url,
        "successful_results": successful_results,
        "failed_usns": failed_usns
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    
    if not Config.API_KEYS:
        print("="*60)
        print("❌ CRITICAL FAILURE: NO GEMINI API KEYS AVAILABLE.")
        print(f"Please define 'GEMINI_API_KEY' or 'GEMINI_API_KEY1-9' in your .env file.")
        print("="*60)
        sys.exit(1)
    
    print("="*60)
    print(f"✅ Successfully loaded {len(Config.API_KEYS)} API key(s).")
    print(f"✅ Concurrency set to {Config.MAX_SCRAPER_WORKERS} workers.")
    print(f"✅ Using Model: {Config.MODEL_NAME}")
    print("Starting Flask server on http://127.0.0.1:5000")
    print("="*60)
    
    app.run(debug=False, host='127.0.0.1', port=5000)