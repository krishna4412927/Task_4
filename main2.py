import boto3
import json
import random
import base64

# Titan → extract key data
# Claude → summarize & classify
# Titan Image Generator

# AWS Bedrock Model IDs 
Text_Model = "amazon.titan-text-express-v1"  
Summarized_Model = "anthropic.claude-3-sonnet-20240229-v1:0"  
Image_Model = "amazon.titan-image-generator-v1" 

# Bedrock runtime client  
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Step 1 — Get user input  
input_text = input(" Enter a product review or any raw text: ").strip()

if not input_text:
    print(" No input provided! Please enter some text.")
    exit()

# Step 2 — Extract key data (Titan)  
prompt_extract = f"""
Extract the following details from the product review below and return JSON with:
product_name, key_features, issues, and emotions.

Review:
{input_text}
"""

titan_request = {
    "inputText": prompt_extract,
    "textGenerationConfig": {"maxTokenCount": 512, "temperature": 0.5}
}

print("\ Extracting key data using Titan...")
titan_response = client.invoke_model(modelId=Text_Model, body=json.dumps(titan_request))
titan_output = json.loads(titan_response["body"].read())
extracted_text = titan_output["results"][0]["outputText"]
print(" Extracted Info:\n", extracted_text)

# Step 3 Summarize + classify sentiment (Claude) 
prompt_summary = f"""
Summarize the following extracted product review data and classify the sentiment 
as Positive, Negative, or Neutral.
Return JSON with keys: summary and sentiment.

Data:
{extracted_text}
"""

summary_request = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": prompt_summary}]}
    ],
}

print(" Summarizing and classifying sentiment using Claude...")
summary_response = client.invoke_model(modelId=Summarized_Model, body=json.dumps(summary_request))
summary_output = json.loads(summary_response["body"].read())
summary_text = summary_output["content"][0]["text"]
print(" Summary and Sentiment:\n", summary_text)

# Step 4 Generate Image (Titan Image Generator) 
prompt_image = f"""
Create an abstract illustration that visually represents the following review summary
and sentiment (Positive, Negative, or Neutral). 
Avoid real brand logos or people.

{summary_text}
"""

print(" Generating visual summary (image)...")

seed = random.randint(0, 2147483647)
image_request = {
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {"text": prompt_image[:500]},
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "quality": "standard",
        "height": 512,
        "width": 512,
        "cfgScale": 8.0,
        "seed": seed,
    },
}

image_response = client.invoke_model(modelId=Image_Model, body=json.dumps(image_request))
image_output = json.loads(image_response["body"].read())
image_data = base64.b64decode(image_output["images"][0])

# Step 5 Save the image 
output_path = "review_visual.png"
with open(output_path, "wb") as f:
    f.write(image_data)

print(f"Workflow complete! Image saved as: {output_path}")
