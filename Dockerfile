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

# Install system deps
RUN yum -y install libsndfile && yum clean all

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model & app
COPY models/ /opt/models/
COPY main.py ./

# Lambda entrypoint
CMD ["main.handler"]
