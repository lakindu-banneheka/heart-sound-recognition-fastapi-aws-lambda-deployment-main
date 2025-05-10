FROM public.ecr.aws/lambda/python:3.9

# Copy model and code
COPY models/ /var/task/models/
COPY app.py /var/task/
COPY requirements.txt /var/task/

# Install dependencies into /var/task
RUN pip3 install -r requirements.txt --target "/var/task"

# Lambda entrypoint
CMD ["app.handler"]
