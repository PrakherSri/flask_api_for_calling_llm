from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from flasgger import Swagger

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React app
Swagger(app)  # Enable Swagger UI

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"

# Replace with your Hugging Face API token

@app.route('/process', methods=['POST'])
def process():
    """
    Call Hugging Face LLM with file content as context and user prompt.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: false
        description: Optional context file
      - in: formData
        name: model
        type: string
        required: true
        description: Hugging Face model name
      - in: formData
        name: token
        type: string
        required: true
        description: Hugging Face Token
      - in: formData
        name: prompt
        type: string
        required: true
        description: User prompt
    responses:
      200:
        description: LLM response
        schema:
          type: object
          properties:
            status:
              type: string
            file_name:
              type: string
            model:
              type: string
            prompt_preview:
              type: string
            huggingface_result:
              type: object
    """
    file = request.files.get('file')
    model = request.form.get('hf_model')
    token = request.form.get('hf_token')
    prompt = request.form.get('prompt')

    if not model or not prompt:
        return jsonify({'status': 'error', 'message': 'Model and prompt are required.'}), 400

    file_content = file.read().decode('utf-8') if file else ""
    context = file_content

    payload = ""
    # Prepare payload for Hugging Face API
    if not context:
        payload = {
            "inputs": {
                "question": prompt
            }
        }
    else:
        payload = {
            "inputs": {
                "question": prompt,
                "context": context
            }
        }
            
    headers = {
        "Authorization": f"Bearer {token}"
    }

    #model = "meta-llama/Llama-3.1-8B"

    hf_url = HUGGINGFACE_API_URL + model
    try:
        response = requests.post(hf_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "answer": result.get("answer", "No answer found"),
                    "score": result.get("score", 0),
                    "start": result.get("start", 0),  
                    "end": result.get("end", 0)
                }
        elif response.status_code == 404:
            return {"error": f"Model '{model}' not found. Try changing to a different model."}
        elif response.status_code == 503:
            return {"error": "Model is loading. Please wait a moment and try again."}
        else:
            return {"error": f"API request failed: {response.status_code} - {response.text}"}
    except Exception as e:
        return jsonify({
            'status': 'failure',
            'error': str(e)
        }), 500

    """ response = {
        'status': 'success',
        'file_name': file.filename if file else 'No file',
        'model': model,
        'prompt_preview': prompt[:50] + '...' if prompt and len(prompt) > 50 else prompt,
        'huggingface_result': hf_result
    }
    return jsonify(response) """


if __name__ == '__main__':
    app.run(debug=True)
