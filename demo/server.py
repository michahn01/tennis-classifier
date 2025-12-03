import http.server
import socketserver
import argparse
import json
import os
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tennis.common.inference import InferenceEngine
from demo.convert_mp4_to_avi import convert_mp4_to_avi

# Global inference engine
inference_engine = None

class PredictionHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                video_path = data.get('video_path')
                
                if not video_path:
                    self.send_error(400, "Missing 'video_path' in request body")
                    return

                if not os.path.exists(video_path):
                    self.send_error(404, f"Video file not found: {video_path}")
                    return

                # Convert if mp4
                if video_path.lower().endswith('.mp4'):
                    print(f"Converting {video_path} to .avi...")
                    convert_mp4_to_avi(video_path)
                    video_path = os.path.splitext(video_path)[0] + '.avi'
                
                print(f"Running inference on {video_path}...")
                predicted_class, probabilities = inference_engine.predict(video_path)
                
                response = {
                    'predicted_class': predicted_class,
                    'probabilities': probabilities
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Endpoint not found")

def main():
    global inference_engine
    
    parser = argparse.ArgumentParser(description="Start Tennis Classifier Inference Server")
    parser.add_argument("--model_type", required=True, help="Type of model (baseline, attention, skeleton, skeleton_attention)")
    parser.add_argument("--model_weights", required=True, help="Path to model weights")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    args = parser.parse_args()
    
    print(f"Initializing Inference Engine with {args.model_type} model from {args.model_weights}...")
    try:
        inference_engine = InferenceEngine(args.model_type, args.model_weights)
    except Exception as e:
        print(f"Failed to initialize Inference Engine: {e}")
        return

    with socketserver.TCPServer(("", args.port), PredictionHandler) as httpd:
        print(f"Serving at port {args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
