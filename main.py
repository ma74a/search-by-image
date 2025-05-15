from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import pandas as pd
import shutil
import os
from with_faiss import SearchByImage

app = FastAPI(title="Image Search API")

# Load the data
df = pd.read_csv("data/the_data.csv")

def get_image_info(img_name):
    image_info = df[df['image_name'] == img_name].iloc[0]
    image_info = image_info.drop("img_link")
    return image_info

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
    # Create a temporary directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the uploaded file temporarily
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Initialize the search object
        obj = SearchByImage()
        obj.load_save_model()
        
        # Perform the search
        similar = obj.search(query_path=temp_path)
        
        # Process results
        results = []
        for img in similar:
            img_name = Path(img).name
            info = get_image_info(img_name)
            results.append({
                "img_name": img_name,
                "img_info": info.to_dict()
            })
        
        return JSONResponse(content={"results": results})
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Search API. Use POST /search endpoint with an image file to search for similar images."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 