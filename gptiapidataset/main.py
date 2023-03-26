import csv
import os
from dotenv import load_dotenv
import json
import openai

# Enable reading from a .env file

load_dotenv()

# Load the api_key and file path to the dataset

openai_api_key = os.getenv("OPENAI_API_KEY")
file_path = os.getenv(r"DATASET_PATH")
openai.api_key = openai_api_key
# Read every row from a csv dataset
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# For each input, create a prompt for GPT

def create_prompt(item):
    # Create a prompt for GPT based on the input data
    title = item['Title']
    email_address = item['EmailAddress']
    City = item['BusinessUnit']
    country = item['Country']
    hire_date = item['StartDate']
    skills = item['Skills']
    level_seniority = item['LevelSeniority']
    project = item['Project']
    return f"Process the following data: {title}, {email_address}, {City}, {country}, {hire_date}, {skills}, {level_seniority}, {project}"

# Call the OpenAI API, and store the result

results = []
for item in data:
    prompt = create_prompt(item)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    results.append({
        'input': item,
        'response': response.choices[0].text.strip()
    })

# Save the results

with open('valconSEEdescriptions.json', 'w') as f:
    json.dump(results, f)