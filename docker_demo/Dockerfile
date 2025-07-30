FROM python

# Setting the working directory within the container
WORKDIR /app

# Copying the current working dir into the /app folder
COPY . /app

# Installing Dependencies
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "demo_L1.py"]
