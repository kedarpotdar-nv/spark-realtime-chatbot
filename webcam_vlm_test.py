#!/usr/bin/env python3
"""
Webcam to VLM (Qwen3-VL) Streaming Test
Streams webcam frames to llama.cpp server running a vision model.

Requirements:
    pip install opencv-python requests pillow

Usage:
    python webcam_vlm_test.py
    
    # Or specify custom server:
    python webcam_vlm_test.py --server http://spark:8080
"""

import cv2
import base64
import json
import threading
import queue
import time
import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO

class WebcamVLMApp:
    def __init__(self, server_url="http://spark:8080"):
        self.server_url = server_url
        self.running = False
        self.analyzing = False
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.response_queue = queue.Queue()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Webcam → Qwen3-VL (llama.cpp)")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1a1a2e")
        
        self.setup_ui()
        self.setup_webcam()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video feed
        left_frame = tk.Frame(main_frame, bg="#1a1a2e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video label
        video_title = tk.Label(left_frame, text="📹 Webcam Feed", 
                              font=("Helvetica", 14, "bold"),
                              fg="#00d4ff", bg="#1a1a2e")
        video_title.pack(pady=(0, 5))
        
        # Video canvas
        self.video_label = tk.Label(left_frame, bg="#0f0f23", 
                                    width=640, height=480)
        self.video_label.pack(pady=5)
        
        # Control buttons
        btn_frame = tk.Frame(left_frame, bg="#1a1a2e")
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="▶ Start Camera", 
                                   command=self.start_camera,
                                   bg="#2ecc71", fg="white", 
                                   font=("Helvetica", 11, "bold"),
                                   padx=15, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", 
                                  command=self.stop_camera,
                                  bg="#e74c3c", fg="white",
                                  font=("Helvetica", 11, "bold"),
                                  padx=15, pady=5, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = tk.Button(btn_frame, text="🔍 Analyze Frame", 
                                     command=self.analyze_current_frame,
                                     bg="#9b59b6", fg="white",
                                     font=("Helvetica", 11, "bold"),
                                     padx=15, pady=5, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.continuous_var = tk.BooleanVar(value=False)
        self.continuous_cb = tk.Checkbutton(btn_frame, text="Continuous Analysis",
                                            variable=self.continuous_var,
                                            bg="#1a1a2e", fg="#adb5bd",
                                            selectcolor="#0f0f23",
                                            font=("Helvetica", 10))
        self.continuous_cb.pack(side=tk.LEFT, padx=10)
        
        # Prompt input
        prompt_frame = tk.Frame(left_frame, bg="#1a1a2e")
        prompt_frame.pack(fill=tk.X, pady=5)
        
        prompt_label = tk.Label(prompt_frame, text="Prompt:", 
                               fg="#adb5bd", bg="#1a1a2e",
                               font=("Helvetica", 10))
        prompt_label.pack(side=tk.LEFT)
        
        self.prompt_entry = tk.Entry(prompt_frame, width=60,
                                     bg="#0f0f23", fg="white",
                                     insertbackground="white",
                                     font=("Helvetica", 11))
        self.prompt_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.prompt_entry.insert(0, "Describe what you see in this image.")
        
        # Right side - Model response
        right_frame = tk.Frame(main_frame, bg="#1a1a2e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Response label
        response_title = tk.Label(right_frame, text="🤖 Qwen3-VL Response", 
                                 font=("Helvetica", 14, "bold"),
                                 fg="#00d4ff", bg="#1a1a2e")
        response_title.pack(pady=(0, 5))
        
        # Response text area
        self.response_text = scrolledtext.ScrolledText(right_frame, 
                                                       wrap=tk.WORD,
                                                       width=50, height=25,
                                                       bg="#0f0f23", fg="#e0e0e0",
                                                       font=("Consolas", 11),
                                                       insertbackground="white")
        self.response_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Click 'Start Camera' to begin.")
        status_bar = tk.Label(right_frame, textvariable=self.status_var,
                             fg="#6c757d", bg="#1a1a2e",
                             font=("Helvetica", 9))
        status_bar.pack(pady=5)
        
        # Server info
        server_label = tk.Label(right_frame, 
                               text=f"Server: {self.server_url}",
                               fg="#6c757d", bg="#1a1a2e",
                               font=("Helvetica", 9))
        server_label.pack()
        
        # Clear button
        clear_btn = tk.Button(right_frame, text="Clear", 
                             command=lambda: self.response_text.delete(1.0, tk.END),
                             bg="#495057", fg="white",
                             font=("Helvetica", 9))
        clear_btn.pack(pady=5)
        
    def setup_webcam(self):
        self.cap = None
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Error: Could not open webcam!")
            return
            
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL)
        self.status_var.set("Camera running...")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        # Start UI update
        self.update_video_display()
        
    def stop_camera(self):
        self.running = False
        self.continuous_var.set(False)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.status_var.set("Camera stopped.")
        
    def video_loop(self):
        """Capture frames in background thread"""
        last_analysis_time = 0
        analysis_interval = 3.0  # seconds between continuous analyses
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                
                # Put frame in queue for display
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
                
                # Continuous analysis
                if self.continuous_var.get() and not self.analyzing:
                    current_time = time.time()
                    if current_time - last_analysis_time > analysis_interval:
                        last_analysis_time = current_time
                        self.analyze_frame_async(frame.copy())
                        
            time.sleep(0.03)  # ~30 FPS
            
    def update_video_display(self):
        """Update video display in main thread"""
        try:
            frame = self.frame_queue.get_nowait()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
            
        # Check for responses
        try:
            response = self.response_queue.get_nowait()
            self.response_text.insert(tk.END, response + "\n\n")
            self.response_text.see(tk.END)
        except queue.Empty:
            pass
            
        if self.running:
            self.root.after(33, self.update_video_display)
            
    def analyze_current_frame(self):
        """Analyze the current frame on button click"""
        if self.current_frame is not None and not self.analyzing:
            self.analyze_frame_async(self.current_frame.copy())
            
    def analyze_frame_async(self, frame):
        """Send frame to VLM in background thread"""
        self.analyzing = True
        self.status_var.set("Analyzing frame...")
        thread = threading.Thread(target=self._send_to_vlm, args=(frame,), daemon=True)
        thread.start()
        
    def _send_to_vlm(self, frame):
        """Send frame to llama.cpp server"""
        try:
            # Convert frame to JPEG base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            prompt = self.prompt_entry.get()
            
            # Try OpenAI-compatible endpoint first
            payload = {
                "model": "qwen3-vl",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 512,
                "stream": False
            }
            
            # Add timestamp to response
            timestamp = time.strftime("%H:%M:%S")
            self.response_queue.put(f"─── {timestamp} ───")
            
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    self.response_queue.put(content)
                    self.root.after(0, lambda: self.status_var.set("Analysis complete."))
                else:
                    self.response_queue.put(f"Unexpected response format: {result}")
            else:
                # Try legacy llama.cpp endpoint
                self._try_legacy_endpoint(img_base64, prompt)
                
        except requests.exceptions.ConnectionError:
            self.response_queue.put(f"❌ Connection error: Cannot reach {self.server_url}")
            self.root.after(0, lambda: self.status_var.set("Connection failed."))
        except Exception as e:
            self.response_queue.put(f"❌ Error: {str(e)}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)[:50]}"))
        finally:
            self.analyzing = False
            
    def _try_legacy_endpoint(self, img_base64, prompt):
        """Try legacy llama.cpp /completion endpoint"""
        try:
            # Some llama.cpp builds use different image format
            payload = {
                "prompt": f"[img-1]{prompt}",
                "image_data": [{"data": img_base64, "id": 1}],
                "n_predict": 512,
                "stream": False
            }
            
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("content", result.get("response", str(result)))
                self.response_queue.put(content)
                self.root.after(0, lambda: self.status_var.set("Analysis complete (legacy API)."))
            else:
                self.response_queue.put(f"❌ Server error: {response.status_code}\n{response.text[:500]}")
                
        except Exception as e:
            self.response_queue.put(f"❌ Legacy endpoint error: {str(e)}")
            
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Webcam to VLM streaming test")
    parser.add_argument("--server", "-s", default="http://spark:8080",
                       help="llama.cpp server URL (default: http://spark:8080)")
    args = parser.parse_args()
    
    app = WebcamVLMApp(server_url=args.server)
    app.run()


if __name__ == "__main__":
    main()

