# PhotoshoppedFaceDetection_API (PurePixel app)

## Project Description

PurePixel (Backend) serves as the server-side application for PurePixel mobile application, an AI-powered tool designed to detect photoshopped regions in facial images. This backend is developed using Django and Django REST framework to handle image analysis requests and return results.

Frontend repository: [Check here](https://github.com/anh-nt24/PhotoshoppedFaceDetection_App) 

## Features

- **Image Analysis:** Accepts image uploads via a RESTful API.
- **Detection:** Pass the image to a semantic segmentation model to detect edited areas on face (using custom UNet model).
- **Visualization:** Returns results with a heatmap highlighting the edited regions.

## Technologies Used

- Django
- Django REST framework
- Pillow (for image handling)
- Torch, Torchvision (for detection model)
- Matplotlib (for highlighting heatmap)

## Project Structure

```plaintext
PurePixel-Backend/
├── api/
│   ├── migrations/
│   │   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── serializers.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── src/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── model/
│   ├── __init__.py
│   ├── v1/
│   │   ├── best_model.pth
│   ├── detector.py
│   ├── UNet.py
│   ├── UNet_elements.py
├── manage.py
├── Pipfile
├── Pipfile.lock
└── README.md
```


## Getting Started

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- pipenv: [Install pipenv](https://pipenv.pypa.io/en/latest/installation.html)

### Installation

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/anh-nt24/PhotoshoppedFaceDetection_API.git
   cd PhotoshoppedFaceDetection_API
    ```

2. **Install Dependencies:**

    ```sh
    pipenv install
    ```

3. **Activate the Virtual Environment:**

    ```sh
    pipenv shell
    ```

4. **Install detection model:**

    ```sh
    cd model
    gdown --folder https://drive.google.com/drive/folders/16adI1BoeAtjjSfF5ZxZ34Vlt4jZg91Rs?usp=sharing

    ```

5. **Run the Development Server:**
    ```sh
    pipenv run python manage.py runserver
    ```

## API Endpoints

### Detect Edits

1. **Endpoint**

- **POST** `/api/detect-regions`

2. **Request Body**

- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `image`: Image file (JPEG or PNG format, maximum size 5MB)

3. **Responses**

- **200 OK:**
  - Returns a PNG image containing the heatmap visualization of detected regions.
  - **Content-Type:** `image/png`
  
- **400 Bad Request:**
  - If the request body is invalid or missing required parameters.
  - **Error JSON example:**
    ```json
    {
      "error": "Unsupported file type. Allowed types are JPEG and PNG."
    }
    ```

- **500 Internal Server Error:**
  - If an unexpected error occurs during image processing or model prediction.
  - **Error JSON example:**
    ```json
    {
      "error": "Error during prediction: <error_message>"
    }
    ```
