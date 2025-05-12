FROM public.ecr.aws/lambda/python:3.10

# System dependencies (fix libsndfile issue)
RUN yum install -y epel-release \
    && yum install -y libsndfile

# Install pip & Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Lambda handler via Mangum
RUN pip install mangum

CMD ["app.handler"]
