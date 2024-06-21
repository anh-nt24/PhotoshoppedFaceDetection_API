from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
from django.http import HttpResponse
import io
from PIL import Image
from model.detector import ImageManipulationDetector

class DetectRegionsView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        detector = ImageManipulationDetector()
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            image = Image.open(image_file)

            # detection
            heatmap = detector.predict(image)
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap)

            buffer = io.BytesIO()
            heatmap_pil.save(buffer, format='PNG') # save the PIL image to the buffer in PNG format
            byte_im = buffer.getvalue()

            return HttpResponse(byte_im, content_type='image/png')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)