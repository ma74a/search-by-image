from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import shutil
import os
from with_faiss import SearchByImage

# Initialize FastAPI app
app = FastAPI(title="Image Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Make sure required directories exist
os.makedirs("temp", exist_ok=True)

# Load the data
try:
    df = pd.read_csv("data/the_data.csv")
except FileNotFoundError:
    print("Warning: data/the_data.csv not found. Some functionality may be limited.")
    # Create an empty DataFrame with expected columns as fallback
    df = pd.DataFrame(columns=["image_name", "img_link"])

def get_image_info(img_name):
    """Get information about an image from the DataFrame"""
    try:
        image_info = df[df['image_name'] == img_name].iloc[0]
        if 'img_link' in image_info:
            image_info = image_info.drop("img_link")
        return image_info
    except (IndexError, KeyError):
        # Return empty dict if image not found
        return {}

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
        similar = obj.search(query_path=temp_path, top_k=40)
        
        # Process results
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
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Search API. Use POST /search endpoint with an image file to search for similar images.",
            "docs_url": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Heroku"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Heroku
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)