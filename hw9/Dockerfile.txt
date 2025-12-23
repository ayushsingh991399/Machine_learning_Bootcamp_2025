FROM agrigorev/model-2025-hairstyle:v1

# Install required packages
RUN pip install pillow onnxruntime numpy

# Copy the lambda function
COPY lambda_function.py .

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
