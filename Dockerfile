# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /usr/src/app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port available to the world outside this container
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "https://qabot-production.up.railway.app/", "--port", "8080"]
# EXPOSE 5000

# # Run app.py when the container launches
# CMD ["python", "./app.py"]