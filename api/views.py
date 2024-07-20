from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
import numpy as np
from django.http import HttpResponse
import io
from PIL import Image
from model.detector import ImageManipulationDetector
from model.UNet import UNet
import dlib

class DetectRegionsView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        model = UNet(3, 1, True)
        detector = ImageManipulationDetector(model)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']

            validation_response = self.validate_image(image_file)
            if validation_response is not None:
                return validation_response

            image = Image.open(image_file)

            # Check if the image contains a face
            face_detected = self.check_for_face(image)

            # No face is detected
            if not face_detected:
                return Response({'error': 'The uploaded image does not contain a human face.', 'face_detected': False}, status=status.HTTP_400_BAD_REQUEST)

            # Detection
            try:
                heatmap = detector.predict(image)
            except RuntimeError as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            buffer = io.BytesIO()
            heatmap.save(buffer, format='PNG')  # save the PIL image to the buffer in PNG format
            byte_im = buffer.getvalue()

            response = HttpResponse(byte_im, content_type='image/png')
            response['face_detected'] = 'true'
            return response
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def validate_image(self, image_file):
        # Check file type
        allowed_file_types = ['image/jpeg', 'image/png']
        if image_file.content_type not in allowed_file_types:
            return Response({'error': 'Unsupported file type. Allowed types are JPEG and PNG.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Check file size (limit to 5MB)
        max_file_size = 5 * 1024 * 1024  # 5 MB
        if image_file.size > max_file_size:
            return Response({'error': 'File size exceeds the limit of 5MB.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Additional
        try:
            image = Image.open(image_file)
            image.verify()
            image = Image.open(image_file)
            image = image.convert('RGB')
        except (IOError, SyntaxError):
            return Response({'error': 'Invalid image file.'}, status=status.HTTP_400_BAD_REQUEST)
    
    def check_for_face(self, image):
        image_np = np.array(image)
        detector = dlib.get_frontal_face_detector()
        faces = detector(image_np, 1)
        return len(faces) > 0

'''
#####################
# API Documentation #
#####################

Endpoint:
POST /api/detect-regions

Request Body:
- Content-Type: multipart/form-data
- Parameters:
  - image: Image file (JPEG or PNG format, maximum size 5MB)

Responses:
- 200 OK:
  - Returns a PNG image containing the heatmap visualization of detected regions.
  - Content-Type: image/png
  
- 400 Bad Request:
  - If the request body is invalid or missing required parameters.
  - Error JSON example:
    {
      "error": "Unsupported file type. Allowed types are JPEG and PNG."
    }

- 500 Internal Server Error:
  - If an unexpected error occurs during image processing or model prediction.
  - Error JSON example:
    {
      "error": "Error during prediction: <error_message>"
    }
'''