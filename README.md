# CrystalClear

## Description

CrystalClear is a web application that allows you to predict the type of gemstone using a CNN (Convolutional Neural Network) model. This model achieves an accuracy rate of 96% thanks to the utilization of TensorFlow technology. Backend is build via Flask, for fronted i choosed Next.js.

# Backend

Backend of application handles:
- Predictions based on images via CNN model

## File structure
- models/ - Containes two best models that i trained for application
- app.py - Flask file that contain logic for handling requests.
- cnn.py - Python file for training CNN model using Tenserflow.
- test.py - Python file used for calculating accuracy of models.

## Technologies Used

- Flask
- Flask_cors
- Python
- Tensorflow
- Numpy

# Frontend

Frontend side of application is build with Next.js. It provides a platform for users to use the functionalities I described in Backend section

## File Structure

- app/contexts - folder that contains context that store results of model prediction.
- app/components - folder for smaller components used for building whole pages.
- public/ - folder that stores images used on frontend.

## Technologies Used

- Next.js
- Typescript
- Axios
- Tailwind CSS
- Framer Motion

  # Instalation Instructions

1. Clone the Repository:
git clone https://github.com/your-account/CrystalClear.git

2. Instal Dependencies:

```bash
cd CrystalClear
cd backend
pip install -r requirements.txt
cd ..
cd frontend
npm install
```

3.Running the Application:
- Backend: Navigate to the backend directory and run the Flask application:
```bash
cd ..
cd backend
python app.py
```

- Fronend: Navigate to the fronend directory and run Next.js server
```bash
cd ..
cd frontend
npm run dev
```

The application will be accessible at http://localhost:3000.

# Using the Application
1. Uploading an Image: On the homepage of the application, you can upload an image of a gemstone you want to identify using the "Choose File" button.
2. Resetting: You can reset the application to upload a new image by clicking the "Reset" button.
3. Viewing Results: After uploading the image, the application will display predictions of the gemstone type.
4. Make new prediction: After reading all the info you can go back and make another prediction by clicking the "Back to Predictions!" button.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License (MIT)




