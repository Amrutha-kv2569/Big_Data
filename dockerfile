# ==============================
#  Dockerfile for Streamlit App
# ==============================

# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all project files to the container
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose Streamlit's default port
EXPOSE 8501

# 6. Environment variables for Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 7. Start the Streamlit app
CMD ["streamlit", "run", "app.py"]
