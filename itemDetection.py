from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import base64

load_dotenv()

app = Flask(__name__)

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OAI_DEPLOYMENT")
azure_cvision_endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")  # Azure Computer Vision endpoint from .env
azure_cvision_key = os.getenv("AZURE_COMPUTER_VISION_KEY")  # Azure Computer Vision key from .env
api_version = '2023-12-01-preview'  # Update this version as necessary

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}/extensions",
)

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    print("Received request to analyze image.")
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        encoded_image = request.json['image']

        # Decode base64 image
        image_data = base64.b64decode(encoded_image.encode())

        # Log image size for verification
        print(f"Image received. Size: {len(image_data)} bytes.")

        # Prepare extra_body for Azure OpenAI request
        extra_body = {
            "dataSources": [
                {
                    "type": "AzureComputerVision",
                    "parameters": {
                        "endpoint": azure_cvision_endpoint,
                        "key": azure_cvision_key
                    }
                }
            ],
            "enhancements": {
                "ocr": {
                    "enabled": True
                },
                "grounding": {
                    "enabled": True
                }
            }
        }

        # Prepare message for Azure OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are an expert image analyst specializing in fashion and accessories."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze the image and identify all fashion items and accessories present. For each item, provide the following attributes separated by commas and different items will be on a different line: Item Type, Color, Pattern, Material, Brand, Size, Additional Details. If any attribute cannot be identified, state 'Not identifiable:'"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]

        print("Sending request to Azure OpenAI with the following data:")
        print(f"Messages: {messages}")
        print(f"Extra Body: {extra_body}")

        # Send request to Azure OpenAI
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            extra_body=extra_body,
            max_tokens=2000
        )

        # Process response from Azure OpenAI
        response_dict = {
            'id': response.id,
            'choices': [
                {
                    'message': {
                        'role': choice.message.role,
                        'content': choice.message.content
                    },
                    'finish_reason': choice.finish_reason,
                    'index': choice.index,
                    'logprobs': choice.logprobs,
                    'content_filter_results': choice.content_filter_results
                } for choice in response.choices
            ],
            'created': response.created,
            'model': response.model,
            'object': response.object,
            'usage': {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }

        print(f"Full response from Azure OpenAI: {response_dict}")

        return jsonify({'response': response_dict})
    
    except Exception as e:
        error_msg = f'Error analyzing image: {str(e)}'
        print(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)
