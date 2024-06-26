FROM mhubai/base:latest

# Set the working directory to /app
WORKDIR /app

# Install system dependencies including curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Download the Code 
RUN git clone https://github.com/FJDorfner/Anatomy-Aware-Classification-axSpA.git

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements_small.txt 
RUN pip install -r /app/Anatomy-Aware-Classification-axSpA/gradio_code/requirements_small.txt && \
    pip uninstall opencv-python -y && \
    pip install opencv-contrib-python-headless==4.5.5.64

# Download the model files from Zenodo using curl
RUN curl -o /app/Anatomy-Aware-Classification-axSpA/gradio_code/classification_model.ckpt -L https://zenodo.org/record/11116990/files/classification_model.ckpt
RUN curl -o /app/Anatomy-Aware-Classification-axSpA/gradio_code/segmentation_model.pt -L https://zenodo.org/record/11116990/files/segmentation_model.pt

# Add code base to python path
ENV PYTHONPATH="/deepspa:/app"

# Default entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/deepspa/config/default.yml"]
