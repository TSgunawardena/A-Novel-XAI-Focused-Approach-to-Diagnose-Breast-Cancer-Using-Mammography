const axios = require('axios');
const FormData = require('form-data');

exports.processImage = async (filePath) => {
  try {
    // Prepare the image to be sent to the ML model API
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));

    // Specify the URL of your FastAPI endpoint
    const mlApiUrl = 'http://localhost:8000/predict';

    // Send the image to the FastAPI server and get the response
    const response = await axios.post(mlApiUrl, formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    // Logging the response from the ML server
    console.log(`Response from ML model: ${JSON.stringify(response.data)}`);

    // Return the response from the FastAPI model
    return {
      fileName: path.basename(filePath),
      prediction: response.data.prediction,
      confidence: response.data.confidence,
      heatmapUrl: response.data.heatmap_url  // assuming FastAPI sends back a URL to a generated heatmap
    };

  } catch (error) {
    console.error("Error processing image with ML model:", error);
    throw new Error("Image processing failed: " + error.message);
  }
};
