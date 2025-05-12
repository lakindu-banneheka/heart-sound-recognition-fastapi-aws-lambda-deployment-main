# FROM public.ecr.aws/lambda/python:3.9

# # Copy model and code
# COPY models/ /var/task/models/
# COPY app.py /var/task/
# COPY requirements.txt /var/task/

# # Install dependencies into /var/task
# RUN pip3 install -r requirements.txt --target "/var/task"

# # Lambda entrypoint
# CMD ["app.handler"]


FROM public.ecr.aws/lambda/python:3.10

# Install libsndfile for soundfile/librosa
RUN yum install -y epel-release && \
    yum install -y libsndfile

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Install Mangum for AWS Lambda compatibility
RUN pip install mangum

# Lambda entrypoint
CMD ["app.handler"]

