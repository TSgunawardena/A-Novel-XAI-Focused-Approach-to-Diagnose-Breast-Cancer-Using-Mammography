import React, { useState } from "react";
import axios from "axios";
import "./ImageUploader.css";

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [limeUrl, setLimeUrl] = useState(null);

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
    setResponse(null);
    setError(null);
    setLimeUrl(null);
  };

  const handleUpload = async () => {
    if (!image) {
      alert("Please select an image to upload!");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    setLoading(true);
    setError(null);
    setResponse(null);
    setLimeUrl(null);

    try {
      const res = await axios.post("http://localhost:5000/api/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResponse(res.data);
      if (res.data.lime_heatmap_url) {
        setLimeUrl(`http://localhost:8000${res.data.lime_heatmap_url}`);
      }
    } catch (err) {
      console.error("Error:", err);
      setError("Error uploading image or getting prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceText = (confidence) => {
    if (confidence == null || isNaN(confidence)) return "Unavailable";
    return `${(confidence * 100).toFixed(2)}%`;
  };

  return (
    <div className="image-uploader">
      <h2>Upload an Image</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="file-input"
      />
      <button onClick={handleUpload} className="upload-button" disabled={loading}>
        {loading ? "Uploading..." : "Upload"}
      </button>

      {response && (
        <div className="response-container">
          <h3>Upload & Prediction Successful!</h3>
          <p><strong>Prediction:</strong> {response.prediction === 0 ? "Benign" : "Malignant"}</p>
          <p><strong>Confidence Score:</strong> {getConfidenceText(response.confidence)}</p>
        </div>
      )}

      {limeUrl && (
        <div className="heatmap-container">
          <h3>LIME Explanation</h3>
          <img src={limeUrl} alt="LIME Heatmap" className="heatmap-image" />
        </div>
      )}

      {error && <p className="error-message">{error}</p>}
    </div>
  );
}

export default ImageUploader;
