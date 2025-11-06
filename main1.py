import boto3
import json
import pandas as pd
from textblob import TextBlob
import os

client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

MODELS = {
    "amazon.titan-text-express-v1": "Titan",
    "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet",
    "mistral.mistral-7b-instruct-v0:2": "Mistral 7B"
}

PROMPT = "Summarize the benefits and drawbacks of electric vehicles in 150 words."
print(MODELS)

# Model invocation 
def invoke_model(model_id, prompt):
    if "anthropic" in model_id:   
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        }
    elif "mistral" in model_id:
        body = {"prompt": prompt, "max_tokens": 300}
    else: 
        body = {"inputText": prompt}

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    if "anthropic" in model_id:
        return result["content"][0]["text"]
    elif "mistral" in model_id:
        return result["outputs"][0]["text"]
    else:
        return result["results"][0]["outputText"]

# Evaluation metrics 
# Uses TextBlob to detect sentiment:(can use textblob in translation also)
# Positive → polarity > 0.1
# Negative → polarity < -0.1
# Neutral → otherwise
def evaluate_response(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.1:
        tone = "Positive"
    elif sentiment < -0.1:
        tone = "Negative"
    else:
        tone = "Neutral"
    return {
        "Response Length": len(text),
        "Sentiment Tone": tone
    }

def main():
    os.makedirs("output", exist_ok=True)
    results = []

    print(" Evaluating multiple AWS Bedrock models...")

    for model_id, model_name in MODELS.items():
        print(f" Invoking: {model_name}")
        try:
            response = invoke_model(model_id, PROMPT)
            metrics = evaluate_response(response)
            metrics["Model"] = model_name
            metrics["Response (first 200 chars)"] = response[:200]
            results.append(metrics)
            print(f" Done: {model_name}")
        except Exception as e:
            print(f" Error with {model_name}: {e}")
            results.append({
                "Model": model_name,
                "Response Length": 0,
                "Sentiment Tone": "Error",
                "Response (first 200 chars)": str(e)
            })

    df = pd.DataFrame(results)
    df.to_csv("output/metrics.csv", index=False)
    print(" Results saved to output/metrics.csv")
    print(df)

if __name__ == "__main__":
    main()
