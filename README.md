# Search By Image

A content-based image retrieval system that finds similar images to a query image using deep learning features and FAISS vector similarity search.

## Features

- **Fast Image Search**: Find visually similar images in milliseconds using FAISS similarity search
- **Deep Learning Based**: Uses ResNet50 pre-trained on ImageNet for feature extraction
- **Easy-to-use API**: Simple REST API with FastAPI for image upload and search
- **Web Interface**: Basic web interface for uploading and searching images
- **Dockerized**: Easy deployment with Docker

## How It Works

1. **Feature Extraction**: The system extracts high-dimensional feature vectors from images using the ResNet50 model
2. **Indexing**: Features are indexed using Facebook AI Similarity Search (FAISS) for efficient similarity retrieval
3. **Similarity Search**: When a query image is uploaded, the system:
   - Extracts features from the query image
   - Searches the FAISS index for similar feature vectors
   - Returns metadata for the most similar images

## System Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│ User Query  │────▶│ Feature       │────▶│ FAISS        │
│ Image       │     │ Extraction    │     │ Similarity   │
└─────────────┘     └───────────────┘     │ Search       │
                                          └──────┬───────┘
                                                 │
                    ┌───────────────┐            │
                    │ Image         │◀───────────┘
                    │ Metadata      │
                    └───────┬───────┘
                            │
┌─────────────┐     ┌──────▼────────┐
│ Web UI or   │◀────│ API Response  │
│ API Client  │     │ with Results  │
└─────────────┘     └───────────────┘
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- PyTorch
- FAISS-CPU (or FAISS-GPU for better performance)
- FastAPI

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/search-by-image.git
   cd search-by-image
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### With Docker

1. Build the Docker image:
   ```bash
   docker build -t search-by-image .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 search-by-image
   ```

## Usage

### Data Preparation

1. Place your images in the `data/the_images` directory
2. Create a CSV file at `data/the_data.csv` with image metadata (optional)

### Building the Index

Run the indexing script to extract features and build the FAISS index:

```bash
python with_faiss.py
```

This will:
- Extract features from all images in the data directory
- Build a FAISS index
- Save the index, features, and image paths to the `saved` directory

### Starting the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- GET `/`: Welcome message and API info
- GET `/search`: Web interface for image upload and search
- POST `/search`: Endpoint for image upload and search
- GET `/health`: Health check endpoint

### Example Usage

1. Open your browser and go to `http://localhost:8000/search`
2. Upload an image using the web interface
3. View similar images returned by the system

## Project Structure

```
search-by-image/
├── config.py           # Configuration parameters
├── dataset.py          # Dataset class for image loading
├── main.py             # FastAPI application
├── with_faiss.py       # FAISS indexing and search implementation
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── data/
│   ├── the_images/     # Directory for storing images
│   └── the_data.csv    # Optional metadata for images
└── saved/
    ├── faiss_index.faiss  # Saved FAISS index
    ├── features.npy       # Extracted features
    └── images_paths.txt   # Paths to indexed images
```

## Model Details

- **Feature Extractor**: ResNet50 pre-trained on ImageNet
- **Feature Dimension**: 2048
- **Similarity Metric**: Inner product (cosine similarity on normalized vectors)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Improvements

- Add support for GPU acceleration
- Implement user authentication
- Add support for image preprocessing options
- Implement support for multiple image collections
- Add more advanced metadata filtering
