const { processImage } = require("../utils/processImage");

exports.uploadImage = async (req, res) => {
  try {
    const filePath = req.file.path;

    // Call the image processing logic
    const result = await processImage(filePath);

    res.status(200).json({
      success: true,
      message: "Image uploaded and processed successfully!",
      data: result,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "An error occurred during image processing.",
      error: error.message,
    });
  }
};
