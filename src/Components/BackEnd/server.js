const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const axios = require("axios");
require("dotenv").config(); // For environment variables

const app = express();
const PORT = process.env.PORT || 5000; // Use PORT from .env or default to 5000

// Middleware
app.use(cors()); // Allow cross-origin requests
app.use(express.json()); // Parse JSON bodies
app.use("/uploads", express.static(path.join(__dirname, "uploads"))); // Serve static files

// Configure Multer for File Uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, "uploads");
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + path.extname(file.originalname); // Unique filenames
    cb(null, uniqueSuffix);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    // Validate file type (e.g., only images)
    const allowedTypes = /jpeg|jpg|png|gif/;
    const extName = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimeType = allowedTypes.test(file.mimetype);

    if (extName && mimeType) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed!"));
    }
  },
});

// API Endpoint for Image Upload
app.post("/api/upload", upload.single("image"), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const filePath = req.file.path;

    return res.json({
      success: true,
      message: "File uploaded successfully",
      filePath: `/uploads/${req.file.filename}`, // Use relative path for access
    });
  } catch (error) {
    console.error("Error uploading file:", error);
    return res.status(500).json({ error: "An error occurred during file upload" });
  }
});

// API Endpoint for ML Model Prediction
const fs = require('fs');
const FormData = require('form-data');

app.post("/api/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded for prediction" });
    }

    const filePath = req.file.path;

    // Create FormData and append the image
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));

    // Send image to FastAPI server
    const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    // Return FastAPI response to client
    res.json(response.data);
  } catch (error) {
    console.error("Error during prediction:", error.message);
    res.status(500).json({ error: "An error occurred during prediction" });
  }
});


// Centralized Error Handling
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err.message);
  res.status(500).json({ error: err.message });
});

// Start the Backend Server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

module.exports = app;
