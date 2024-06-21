from django.urls import path
from .views import DetectRegionsView

urlpatterns = [
    path(route='detect-regions/', view=DetectRegionsView.as_view(), name='detect-regions'),
]