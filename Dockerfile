# FROM public.ecr.aws/lambda/python:3.9

# # Copy model and code
# COPY models/ /var/task/models/
# COPY app.py /var/task/
# COPY requirements.txt /var/task/

# # Install dependencies into /var/task
# RUN pip3 install -r requirements.txt --target "/var/task"

# # Lambda entrypoint
# CMD ["app.handler"]


FROM public.ecr.aws/lambda/python:3.9

# Install system dependency for soundfile
RUN yum -y install libsndfile && \
    yum clean all

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model to /opt/models
COPY models/audio_model.keras /opt/models/audio_model.keras

# Copy our FastAPI app
COPY main.py ./

# Set the Lambda handler
# "main.handler" points to the Mangum handler in main.py
CMD ["main.handler"]
