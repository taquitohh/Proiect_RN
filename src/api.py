import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:5173'], supports_credentials=True)

# Global model instance
code_generator = None

def load_code_generator():
    global code_generator
    try:
        from neural_network.code_generator_model import CodeGeneratorInference
        root_dir = Path(__file__).parent.parent
        model_path = str(root_dir / 'models' / 'code_generator.pt')
        vocab_path = str(root_dir / 'config' / 'code_vocab.pkl')
        
        if not os.path.exists(model_path):
            return False
            
        code_generator = CodeGeneratorInference(model_path, vocab_path)
        return code_generator.is_loaded
    except Exception as e:
        print(f'Error: {e}')
        return False

# Load model at startup
load_code_generator()

@app.route('/')
def root():
    return jsonify({'message': 'Pure AI Blender API', 'status': 'online'})

@app.route('/api/status')
def get_status():
    return jsonify({
        'pure_ai_mode': True, 
        'model_loaded': code_generator.is_loaded if code_generator else False
    })

@app.route('/api/blender/generate', methods=['POST'])
def generate_blender_code():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text'}), 400
        
        user_text = data['text'].strip()
        if not code_generator or not code_generator.is_loaded:
            return jsonify({'success': False, 'error': 'MODEL_NOT_LOADED'}), 503
        
        # MODEL INFERENCE - This is where the AI generates the code
        result = code_generator.generate(text=user_text, temperature=0.5)
        
        if not result['success']:
            return jsonify({'success': False, 'error': result.get('error')}), 500
            
        return jsonify({
            'success': True,
            'code': result['code'],
            'method': 'neural_network_seq2seq',
            'interpretation': 'Generat integral de Rețeaua Neuronală'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/blender/model-status')
def get_model_status():
    if code_generator:
        return jsonify({'loaded': code_generator.is_loaded})
    return jsonify({'loaded': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
