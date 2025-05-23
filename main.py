from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import shutil
import os
import traceback

from with_faiss import SearchByImage
from config import Config

app = FastAPI(title="Image Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

os.makedirs("temp", exist_ok=True)

try:
    df = pd.read_csv("data/the_data.csv")
except FileNotFoundError:
    print("Warning: data/the_data.csv not found. Some functionality may be limited.")
    df = pd.DataFrame(columns=["image_name", "img_link"])

def get_image_info(img_name):
    try:
        image_info = df[df['image_name'] == img_name].iloc[0]
        if 'img_link' in image_info:
            image_info = image_info.drop("img_link")
        return image_info
    except (IndexError, KeyError):
        return {}

# Initialize SearchByImage once at startup
search_obj = SearchByImage()

# Verify required files exist before loading
for path in [Config.INDEX_PATH, Config.FEATURES_PATH, Config.IMAGES_PATHS]:
    if not os.path.exists(path):
        print(f"Warning: Required file {path} not found!")

try:
    search_obj.load_save_model()
    print("Search model loaded successfully")
except Exception as e:
    print(f"Error loading search model: {e}")
    traceback.print_exc()

@app.get("/search", response_class=HTMLResponse)
async def search_page():
    return """
    <html>
        <head>
            <title>Image Search</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
                .upload-form:hover { border-color: #666; }
                .submit-btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                .submit-btn:hover { background-color: #45a049; }
            </style>
        </head>
        <body>
            <h1>Image Search</h1>
            <div class="upload-form">
                <form action="/search" method="post" enctype="multipart/form-data">
                    <h2>Upload an image to find similar images</h2>
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <input type="submit" value="Search" class="submit-btn">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        similar = search_obj.search(query_path=temp_path, top_k=40)

        results = []
        for img in similar:
            img_name = Path(img).name
            info = get_image_info(img_name)
            results.append({
                "img_name": img_name,
                "img_info": info.to_dict() if not isinstance(info, dict) else info
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Search API. Use POST /search endpoint with an image file to search for similar images.",
            "docs_url": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Heroku
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)