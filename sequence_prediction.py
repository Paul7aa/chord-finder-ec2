from uuid import uuid4
from fastapi import FastAPI
from celery_worker import process_audio
from celery_worker import celery_app

app = FastAPI()

@app.post("/analyse/")
async def analyse(audio_data_json: dict):
    try:
        task_id = str(uuid4())  # Generate a unique task ID
        process_audio.apply_async(args=[audio_data_json], task_id=task_id)
        return {"task_id": task_id}
    except:
        print("Failed to process audio")
        return "Failed to process audio"
    
    
@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = celery_app.AsyncResult(task_id)
    if task.state == "PENDING":
        response = {
            "state": task.state,
            "status": "Task is still processing"
        }
    elif task.state != "FAILURE":
        response = {
            "state": task.state,
            "result": task.result,
        }
    else:
        response = {
            "state": task.state,
            "status": str(task.info)
        }
    return response