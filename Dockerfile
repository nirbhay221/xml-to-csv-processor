FROM python:3.9-slim

# Creating a directory for the application
WORKDIR /app

# Copying requirements.txt first for better layer caching
COPY requirements.txt /app/

# Installing required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copying the script to the container
COPY xml_to_csv.py /app/

# Creating entrypoint script that preserves the original behavior
RUN echo '#!/bin/bash\n\
input_file=$1\n\
input_dir=$(dirname "$input_file")\n\
filename=$(basename "$input_file")\n\
cd "$input_dir"\n\
python /app/xml_to_csv.py "$filename"\n' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Setting the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]