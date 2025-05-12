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

# Install libsndfile (needed by soundfile/librosa)
RUN yum install -y epel-release && \
    yum install -y libsndfile

# Set working directory
WORKDIR /var/task

# Copy application files
COPY models/ ./models/
COPY app.py .
COPY requirements.txt .

# Install Python dependencies into /var/task
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --target .

# Lambda entrypoint
CMD ["app.handler"]
