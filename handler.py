import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def serverless_pipeline(model_path='./model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    def summary(article):
        input_ids = tokenizer(article, return_tensors="pt").input_ids
        output = model.generate(
            input_ids, 
            max_length=50, 
            num_beams=5, 
            early_stopping=True
        )
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return(summary)
    return(summary)

# initializes the pipeline
summarization_pipeline = serverless_pipeline()

def handler(event, context):
    try:
        print(event)
        print(context)
        # loads the incoming event into a dictonary
        body = json.loads(event['body'])
        # uses the pipeline to predict the answer
        answer = summarization_pipeline(article=body['article'])
        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'answer': answer})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }