# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory to /app
# WORKDIR /src/appFlask

# Copy the current directory contents into the container at /app
COPY src /app
COPY setup.py setup.py
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir .

# Make port 5000 available to the world outside this container
EXPOSE 5001

# Define environment variable
#

# Run app.py when the container launches
ENTRYPOINT ["python3", "/app/app.py"]
CMD ["echo", "Hello"]
