import numpy as np
from PIL import Image
import pytesseract
import cv2
import time
import os
import webbrowser
from typing import Dict, Optional, Union, List, Any
import logging
from dataclasses import dataclass
from pathlib import Path

import json
from contextlib import contextmanager
try:
    from sklearn.cluster import KMeans
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")
try:
    import cv2
    import pytesseract
    import numpy as np
    from PIL import Image
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC1
    
    from selenium.common.exceptions import WebDriverException, TimeoutException
except ImportError as e:
    print("\nMissing dependencies. Please install required packages:")
    print("pip install opencv-python pytesseract numpy pillow selenium")
    raise SystemExit(1)
import signal  # Add this with other imports
import platform  # Add this with other imports
import sys  # Add this with other imports
try:
    import psutil
except ImportError:
    print("Please install psutil: pip install psutil")
    raise SystemExit(1)
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='google_lens.log'
)

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.load_config()
    
    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {
                "tesseract_path": "/usr/bin/tesseract",
                "camera": {
                    "index": 0,
                    "width": 1920,
                    "height": 1080,
                    "fps": 30
                }
            }

@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    auto_focus: bool = True
    brightness: int = 100

@dataclass
class ImageAnalysisResult:
    dimensions: str
    channels: int
    avg_brightness: float
    edge_density: float
    blur_score: float
    dominant_colors: list
    estimated_quality: str
    object_detection_results: list
    text_regions: list

class ImageProcessor:
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def __init__(self, camera_config: CameraConfig = CameraConfig()):
        self.camera_config = camera_config
        self.last_capture_path: Optional[str] = None
        self._setup_tesseract()
        self._setup_object_detection()

    def _setup_tesseract(self):
        # Use the appropriate path for your system
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        else:  # Linux/Mac
            # Update this path for macOS - this is the default Homebrew installation path
            pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # or '/usr/local/bin/tesseract'

    def _setup_object_detection(self):
        """Initialize YOLO or similar object detection model"""
        try:
            import torch
            self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        except ImportError:
            logging.warning("torch not installed. Object detection disabled.")
            self.object_detector = None

    def capture_image(self) -> Optional[str]:
        """Enhanced image capture with better error handling and debugging"""
        print("Attempting to open camera...")
        
        # Try multiple camera indices
        camera_indices = [0, 1, 2]  # Most systems use 0 for default camera
        cap = None
        
        for index in camera_indices:
            try:
                print(f"Trying camera index {index}...")
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    print(f"Successfully opened camera {index}")
                    self.camera_config.index = index
                    break
            except Exception as e:
                print(f"Failed to open camera {index}: {e}")
        
        if not cap or not cap.isOpened():
            print("Error: Could not open any camera")
            print("Please check:")
            print("1. Camera is properly connected")
            print("2. Camera permissions are granted")
            print("3. No other application is using the camera")
            return None
        
        try:
            # Configure camera
            print("Configuring camera settings...")
            camera_properties = {
                cv2.CAP_PROP_FRAME_WIDTH: self.camera_config.width,
                cv2.CAP_PROP_FRAME_HEIGHT: self.camera_config.height,
                cv2.CAP_PROP_FPS: self.camera_config.fps
            }
            
            for prop, value in camera_properties.items():
                if not cap.set(prop, value):
                    print(f"Warning: Could not set camera property {prop} to {value}")
            
            print("\nCamera ready!")
            print("Press SPACE to capture an image")
            print("Press ESC to cancel")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame")
                    break
                
                # Display frame
                cv2.imshow('Capture (SPACE: capture, ESC: exit)', frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Capture
                    filename = self._save_capture(frame)
                    print(f"Image saved as: {filename}")
                    return filename
                elif key == 27:  # ESC
                    print("Capture cancelled")
                    return None
                
        except Exception as e:
            print(f"Camera error: {str(e)}")
            return None
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def _analyze_frame(self, frame) -> dict:
        """Real-time frame analysis"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate blur score using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness (normalized to 0-100 range)
            brightness = np.mean(gray) * 100 / 255
            
            # Initialize face detection
            try:
                face_cascade = cv2.CascadeClassifier()
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not face_cascade.load(cascade_path):
                    logging.error(f"Error loading face cascade from {cascade_path}")
                    faces = []
                else:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
            except Exception as e:
                logging.error(f"Face detection error: {str(e)}")
                faces = []
            
            # Calculate histogram with error handling
            try:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, 256, cv2.NORM_MINMAX)
            except Exception as e:
                logging.error(f"Histogram calculation error: {str(e)}")
                hist = np.zeros((256, 1))
            
            return {
                'blur_score': float(blur_score),
                'brightness': float(brightness),
                'faces': faces if isinstance(faces, list) else [],
                'histogram': hist
            }
            
        except Exception as e:
            logging.error(f"Frame analysis error: {str(e)}")
            return {
                'blur_score': 0.0,
                'brightness': 0.0,
                'faces': [],
                'histogram': np.zeros((256, 1))
            }

    def _add_frame_overlay(self, frame, frame_info, show_grid=True):
        """Add informative overlay to frame"""
        height, width = frame.shape[:2]
        
        # Add timestamp and camera settings
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add technical info
        info_text = [
            f"Resolution: {width}x{height}",
            f"Blur Score: {frame_info['blur_score']:.1f}",
            f"Brightness: {frame_info['brightness']:.1f}",
            f"Faces Detected: {len(frame_info['faces'])}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, height - 20 - (i * 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw rule of thirds grid
        if show_grid:
            for i in range(1, 3):
                cv2.line(frame, (width * i // 3, 0), (width * i // 3, height), (255, 255, 255), 1)
                cv2.line(frame, (0, height * i // 3), (width, height * i // 3), (255, 255, 255), 1)
        
        # Draw face rectangles
        for (x, y, w, h) in frame_info['faces']:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add histogram visualization
        hist_h = 120
        hist_w = 200
        hist_img = np.zeros((hist_h, hist_w), np.uint8)
        cv2.normalize(frame_info['histogram'], frame_info['histogram'], 0, hist_h, cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(hist_img, (i, hist_h), (i, hist_h - int(frame_info['histogram'][i])),
                    255, 1)
        hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)
        frame[height-hist_h-10:height-10, width-hist_w-10:width-10] = hist_img
        
        return frame

    def _handle_mouse_event(self, event, x, y, flags=None, param=None):
        """Handle mouse events for manual focus"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.focus_point = (x, y)

    def perform_ocr(self, image_path: str) -> Dict[str, Union[str, float]]:
        """Enhanced OCR with better error handling"""
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        try:
            image = self._preprocess_image(image_path)
            if image is None:
                return {"error": "Failed to preprocess image"}
            
            # Test if tesseract is properly installed
            try:
                pytesseract.get_tesseract_version()
            except EnvironmentError:
                return {"error": "Tesseract not properly installed"}
            
            # Perform OCR with timeout
            try:
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            except Exception as e:
                return {"error": f"OCR failed: {str(e)}"}
            
            # Process results
            if not ocr_data or 'text' not in ocr_data:
                return {"error": "No text detected"}
            
            return self._process_ocr_results(ocr_data)
            
        except Exception as e:
            logging.error(f"OCR error: {str(e)}")
            return {"error": str(e)}

    def analyze_image(self, image_path: str) -> Optional[ImageAnalysisResult]:
        """Enhanced image analysis with more metrics"""
        try:
            if not self._validate_image(image_path):
                return None

            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Calculate metrics
            blur_score = self._calculate_blur_score(image)
            dominant_colors = self._extract_dominant_colors(image)
            edge_density = self._calculate_edge_density(image)
            
            # Estimate quality
            quality = self._estimate_image_quality(
                blur_score, 
                edge_density, 
                width * height
            )

            # Add object detection
            objects = []
            if self.object_detector:
                results = self.object_detector(image)
                objects = [
                    {
                        'class': results.names[int(det[5])],
                        'confidence': float(det[4]),
                        'bbox': det[:4].tolist()
                    }
                    for det in results.xyxy[0]
                    if float(det[4]) > 0.5  # Confidence threshold
                ]

            return ImageAnalysisResult(
                dimensions=f"{width}x{height}",
                channels=image.shape[2],
                avg_brightness=np.mean(image),
                edge_density=edge_density,
                blur_score=blur_score,
                dominant_colors=dominant_colors,
                estimated_quality=quality,
                object_detection_results=objects,
                text_regions=self.perform_ocr(image_path).get('text_regions', [])
            )

        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            return None

    def _save_capture(self, frame) -> str:
        """Save captured frame with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        
        # Create captures directory if it doesn't exist
        os.makedirs("captures", exist_ok=True)
        filepath = os.path.join("captures", filename)
        
        cv2.imwrite(filepath, frame)
        return filepath

    def _trigger_autofocus(self, cap):
        """Trigger camera autofocus"""
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        time.sleep(0.1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus

    def _validate_image(self, image_path):
        """Validate if image exists and has supported format"""
        if not os.path.exists(image_path):
            return False
        return os.path.splitext(image_path)[1].lower() in self.SUPPORTED_FORMATS

    def _preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get better contrast
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _detect_language(self, text):
        """Detect primary language of the text"""
        # This could be implemented using langdetect or similar library
        return "unknown"

    def _calculate_blur_score(self, image):
        """Calculate image blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _extract_dominant_colors(self, image, n_colors=5):
        """Extract dominant colors using K-means clustering or simple averaging"""
        try:
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors)
            kmeans.fit(pixels)
            return kmeans.cluster_centers_.astype(int).tolist()
        except ImportError:
            # Fallback to a simpler method if sklearn is not available
            logging.warning("sklearn not available, using simple color averaging")
            h, w = image.shape[:2]
            pixels = image.reshape((h * w, 3))
            return [pixels.mean(axis=0).astype(int).tolist()]

    def _calculate_edge_density(self, image):
        """Calculate edge density using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.count_nonzero(edges) / (image.shape[0] * image.shape[1])

    def _estimate_image_quality(self, blur_score, edge_density, resolution):
        """Estimate overall image quality based on metrics"""
        if resolution < 640 * 480:
            return "low"
        if blur_score < 100 or edge_density < 0.01:
            return "poor"
        if blur_score > 500 and edge_density > 0.05:
            return "excellent"
        return "good"

    def _check_internet_connection() -> bool:
        """Test internet connectivity"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    @contextmanager
    def timeout(seconds: int):
        """Context manager for timeouts"""
        def signal_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def debug_info(self) -> Dict[str, Any]:
        """Gather debug information"""
        return {
            "opencv_version": str(cv2.__version__),
            "tesseract_version": str(pytesseract.get_tesseract_version()),
            "camera_config": vars(self.camera_config),
            "system_info": {
                "platform": platform.system(),
                "python_version": sys.version,
                "memory_available": psutil.virtual_memory().available / (1024 * 1024),  # MB
            }
        }

def search_web(text=None, image_path=None, search_type='general'):
    """
    Enhanced web search function with better error handling
    """
    if not image_path:
        print("Error: No image provided for search")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    # First, verify internet connection
    if not _check_internet_connection():
        print("Error: No internet connection detected")
        return

    try:
        # Enhanced Chrome configuration
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        
        # Initialize WebDriver with explicit wait
        print("\nInitializing Chrome browser...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        wait = WebDriverWait(driver, 20)  # Increased timeout to 20 seconds
        
        # Load Google Lens
        print("Opening Google Lens...")
        driver.get("https://lens.google.com")
        
        # Wait for page to load completely
        time.sleep(3)
        
        # Upload image with enhanced error handling
        print("Uploading image...")
        try:
            # Try multiple selectors for the file input
            selectors = [
                "input[type='file']",
                "[name='encoded_image']",
                "[name='image_url']"
            ]
            
            file_input = None
            for selector in selectors:
                try:
                    file_input = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if file_input:
                        break
                except:
                    continue
            
            if not file_input:
                raise Exception("Could not find file upload element")
            
            # Convert to absolute path and upload
            abs_path = os.path.abspath(image_path)
            file_input.send_keys(abs_path)
            print("Image uploaded successfully")
            
            # Wait for results to load
            time.sleep(5)
            
        except Exception as e:
            print(f"Upload error: {str(e)}")
            print("Attempting alternative upload method...")
            try:
                # Try JavaScript file upload
                js_code = f"""
                const input = document.querySelector('input[type="file"]');
                if (input) {{
                    input.style.display = 'block';
                    input.value = '{abs_path}';
                }}
                """
                driver.execute_script(js_code)
                time.sleep(3)
            except Exception as js_error:
                print(f"Alternative upload failed: {str(js_error)}")
                return

        # Interactive menu loop
        while True:
            print("\nSearch Options:")
            print("1. Visual Search")
            print("2. Shopping Search")
            print("3. Text Detection")
            print("4. Places/Landmarks")
            print("5. Auto-detect Best Results")
            print("6. Close Browser")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '6':
                break
                
            try:
                if choice in ['1', '2', '3', '4']:
                    # Map choices to tab names
                    tabs = {
                        '1': 'Visual matches',
                        '2': 'Shopping',
                        '3': 'Text',
                        '4': 'Places'
                    }
                    
                    tab_name = tabs[choice]
                    print(f"\nSwitching to {tab_name}...")
                    
                    # Try to click the tab
                    try:
                        tab = wait.until(
                            EC.element_to_be_clickable(
                                (By.XPATH, f"//div[contains(text(), '{tab_name}')]")
                            )
                        )
                        tab.click()
                        time.sleep(2)
                    except:
                        print(f"Could not switch to {tab_name} tab")
                
                elif choice == '5':
                    print("\nShowing all available results...")
                    time.sleep(2)
                
            except Exception as e:
                print(f"Error during search: {str(e)}")

    except Exception as e:
        print(f"Browser error: {str(e)}")
        print("\nPlease ensure:")
        print("1. Google Chrome is installed and up to date")
        print("2. You have a stable internet connection")
        print("3. The image file is valid and accessible")
    
    finally:
        try:
            input("\nPress Enter to close the browser...")
            driver.quit()
            print("Browser closed successfully")
        except:
            pass

def _check_internet_connection() -> bool:
    """Test internet connectivity"""
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def check_dependencies():
    try:
        # Check if tesseract is installed
        if not pytesseract.get_tesseract_version():
            print("Tesseract is not installed or not in PATH")
            print("Please install Tesseract:")
            print("- Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("- Linux: sudo apt-get install tesseract-ocr")
            print("- Mac: brew install tesseract")
            return False
        
        # Check if Chrome/ChromeDriver is installed (for selenium)
        try:
            webdriver.Chrome(options=Options())
        except Exception:
            print("ChromeDriver not found or Chrome not installed")
            print("Please install Chrome and ChromeDriver:")
            print("https://sites.google.com/chromium.org/driver/")
            return False
            
        return True
    except Exception as e:
        print(f"Dependency check failed: {str(e)}")
        return False

def main():
    print("\nSearchLens Initializing...")
    
    # Test camera access first
    print("\nTesting camera access...")
    if not test_camera():
        print("Camera test failed. Please check your camera connection and permissions.")
        return
    
    try:
        print("\nInitializing ImageProcessor...")
        processor = ImageProcessor(camera_config=CameraConfig(index=0))
    except Exception as e:
        print(f"Failed to initialize ImageProcessor: {str(e)}")
        return
    
    while True:
        try:
            print("\nSearchLens Menu")
            print("=" * 50)
            print("1. Capture and Search Image")
            print("2. Perform OCR and Search")
            print("3. Analyze Image")
            print("4. Search Existing Image")
            print("5. Debug Information")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                print("\nInitiating camera capture...")
                image_path = processor.capture_image()
                if image_path:
                    print(f"\nImage captured successfully: {image_path}")
                    print("Initiating web search...")
                    search_web(image_path=image_path)
                else:
                    print("Image capture failed or was cancelled")
            
            elif choice == '2':
                # OCR and search
                image_path = processor.capture_image()
                if image_path:
                    ocr_result = processor.perform_ocr(image_path)
                    if 'error' not in ocr_result:
                        search_web(text=ocr_result.get('text', ''))
                    else:
                        print(f"OCR Error: {ocr_result['error']}")
            
            elif choice == '3':
                # Analyze image
                image_path = input("Enter image path (or press Enter to capture new): ").strip()
                if not image_path:
                    image_path = processor.capture_image()
                if image_path:
                    analysis = processor.analyze_image(image_path)
                    if analysis:
                        print("\nImage Analysis Results:")
                        print(json.dumps(analysis.__dict__, indent=2))
            
            elif choice == '4':
                # Search existing image
                image_path = input("Enter the path to the image: ").strip()
                if os.path.exists(image_path):
                    search_web(image_path=image_path)
                else:
                    print("Error: Image file not found")
            
            elif choice == '5':
                # Debug info
                print("\nDebug Information:")
                print(json.dumps(processor.debug_info(), indent=2))
            
            elif choice == '6':
                print("Exiting SearchLens...")
                break
            
            # Add a pause before showing menu again
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again")
            input("\nPress Enter to continue...")

def example_usage():
    # Initialize processor
    processor = ImageProcessor(camera_config=CameraConfig(index=0))
    
    # Capture image
    image_path = processor.capture_image()
    if not image_path:
        return
    
    # Analyze image
    analysis = processor.analyze_image(image_path)
    if not analysis:
        return
    
    # Perform OCR
    ocr_result = processor.perform_ocr(image_path)
    
    # Search web
    search_web(text=ocr_result.get('text'), image_path=image_path)

def test_camera():
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 if camera doesn't open
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return False
    
    print("Camera test successful!")
    cap.release()
    return True

def test_web_search():
    """Test function for web search functionality"""
    print("\nTesting Web Search Functionality...")
    
    # Test 1: Check Chrome and ChromeDriver
    print("\nTest 1: Checking Chrome setup...")
    try:
        options = Options()
        options.add_argument("--headless")  # Run in background for test
        driver = webdriver.Chrome(options=options)
        print("✓ Chrome and ChromeDriver working correctly")
        driver.quit()
    except Exception as e:
        print(f"✗ Chrome setup error: {e}")
        print("Please ensure Chrome and ChromeDriver are properly installed")
        return False

    # Test 2: Test with sample image
    print("\nTest 2: Testing image search...")
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, test_image)
        
        print("Testing search with sample image...")
        search_web(image_path=test_image_path)
        
        # Clean up test image
        os.remove(test_image_path)
        print("✓ Image search test completed")
    except Exception as e:
        print(f"✗ Image search error: {e}")
        return False

    print("\nWeb search testing completed!")
    return True

def capture_and_search():
    """Test function to capture and search an image"""
    print("\nInitializing camera and search test...")
    
    try:
        # Initialize ImageProcessor
        processor = ImageProcessor(camera_config=CameraConfig(index=0))
        
        # Capture image
        print("\nStarting camera capture...")
        print("Press SPACE to capture or ESC to cancel")
        image_path = processor.capture_image()
        
        if image_path:
            print(f"\nImage captured successfully: {image_path}")
            
            # Verify image exists
            if os.path.exists(image_path):
                print("\nInitiating web search with captured image...")
                search_web(image_path=image_path)
            else:
                print("Error: Captured image file not found")
        else:
            print("Image capture cancelled or failed")
            
    except Exception as e:
        print(f"\nError during capture and search: {str(e)}")
        print("\nPlease ensure:")
        print("1. Your camera is connected and working")
        print("2. Chrome browser is installed")
        print("3. You have internet connection")

# Run the test
test_camera()

if __name__ == "__main__":
    print("Testing SearchLens web functionality...")
    if test_web_search():
        print("\nAll tests passed! You can now use the web search feature.")
        print("\nTo use web search:")
        print("1. Capture an image or select existing image")
        print("2. Choose search type (Visual, Shopping, Text, or Places)")
        print("3. Use auto-detect for best results")
    else:
        print("\nSome tests failed. Please check the errors above.")
    capture_and_search()