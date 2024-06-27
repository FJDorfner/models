FROM mhubai/base:latest

# Set the working directory to /app
WORKDIR /app

# Install system dependencies including curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Download the Code 
RUN git clone https://github.com/FJDorfner/Anatomy-Aware-Classification-axSpA.git

# Copy the current directory contents into the container at /app
COPY . /app

# Modify requirements_small.txt to remove version constraint for numpy, this is necessary to run in the mhub image.
RUN sed -i '/numpy/d' /app/Anatomy-Aware-Classification-axSpA/gradio_code/requirements_small.txt && \
    echo "\nnumpy" >> /app/Anatomy-Aware-Classification-axSpA/gradio_code/requirements_small.txt

# Install any needed packages specified in requirements_small.txt 
RUN pip install --no-cache-dir -r /app/Anatomy-Aware-Classification-axSpA/gradio_code/requirements_small.txt && \
    pip uninstall opencv-python -y && \
    pip install --no-cache-dir opencv-contrib-python-headless==4.5.5.64 
    
# Download the model files from Zenodo using curl
RUN curl -o /app/classification_model.ckpt -L https://zenodo.org/record/11116990/files/classification_model.ckpt
RUN curl -o /app/segmentation_model.pt -L https://zenodo.org/record/11116990/files/segmentation_model.pt

# Add code base to python path
ENV PYTHONPATH="/app/Anatomy-Aware-Classification-axSpA/gradio_code"

# Default entrypoint
ENTRYPOINT ["mhubio.run"]
CMD ["--config", "/app/models/deepspa/config/default.yml"]
