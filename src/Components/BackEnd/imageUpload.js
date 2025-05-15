const express = require("express");
const multer = require("multer");
const { processImage } = require("../utils/processImage");
const uploadController = require("../controllers/uploadController");

const router = express.Router();

// Multer Configuration for File Uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // Save files to 'uploads' directory
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname); // Unique file name
  },
});

const upload = multer({ storage });

// Route for uploading and processing the image
router.post("/upload", upload.single("image"), uploadController.uploadImage);

module.exports = router;
