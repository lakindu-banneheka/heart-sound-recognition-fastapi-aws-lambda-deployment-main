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

# Install system dependencies required by librosa and soundfile
RUN yum install -y tar gzip make gcc-c++ && \
    yum install -y libsndfile && \
    yum clean all

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code
COPY app.py ${LAMBDA_TASK_ROOT}
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Set the CMD to your handler
CMD ["app.handler"]