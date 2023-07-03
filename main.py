
from fastapi import FastAPI, Body
import uvicorn 
from pydantic import BaseModel
from ultralytics import YOLO
import time
from datetime import datetime

## load model

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


## fast api
app = FastAPI()
class InferenceInput(BaseModel):
    path_image: str
    no: str

def round_time(timex):
    return str(timex).split()[-1].split(".")[0]

@app.post('/inference_y8')
def inference_y8(params: InferenceInput):
    print("***** ***** ***** ***** ***** ***** *****")
    time_start=datetime.now()
    # print(f"==> time start: {time_start}")
    # print(f"path_image: {path_image.path_image}")
    
    ## predict
    results = model.predict(source=params.path_image, 
    conf=0.25)
    # time.sleep(20)
    time_end=datetime.now()

    log=f"\n\t==> time_start {params.no}: \t{round_time(time_start)} ******************\n\
        ==> time_end: \t\t{round_time(time_end)} \n\
        ==> duration: \t\t{round_time(time_end-time_start)}"
    print(log)
    return log

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)

