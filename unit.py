import requests
from datetime import datetime
import sys
def round_time(timex):
    return str(timex).split()[-1].split(".")[0]

## Set the URL of the FastAPI endpoint
url = 'http://0.0.0.0:8000/inference_y8'
print(f"\n==> time_start {sys.argv[1]}: \t{round_time(datetime.now())}")


# Set the string parameter to the path of the file you want to process
file_path = 'dog.jpeg'

# Set the payload for the request
payload = {"path_image": file_path, "no": sys.argv[1]}

# Send the POST request with the payload
response = requests.post(url, json=payload)

# Print the response
# print(response.text)