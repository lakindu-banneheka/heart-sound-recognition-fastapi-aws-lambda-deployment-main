# Use a flexible base image with apt
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    curl \
    && apt-get clean

# Set workdir
WORKDIR /var/task

# Install Lambda Runtime Interface Emulator (RIE)
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/local/bin/aws-lambda-rie
RUN chmod +x /usr/local/bin/aws-lambda-rie

# Copy files
COPY models/ ./models/
COPY app.py .
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Use RIE for Lambda compatibility
ENTRYPOINT ["/usr/local/bin/aws-lambda-rie", "python", "-m", "awslambdaric"]
CMD ["app.handler"]
