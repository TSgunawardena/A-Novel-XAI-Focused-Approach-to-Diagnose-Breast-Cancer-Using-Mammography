import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Home from "./Components/FrontEnd/home/home";
import ImageUploader from "./Components/FrontEnd/ImageUploader/ImageUploader";

function App() {
  return (
    <Router>
      <div className="app-container">
        {/* Navigation Bar */}
        <header className="app-header">
          <nav>
            <ul className="nav-links">
              <li>
                <Link to="/">Home</Link>
              </li>
            </ul>
          </nav>
        </header>

        {/* Main Content */}
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<ImageUploader />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="app-footer">
          <p>&copy; 2025 ML Image Processor. All Rights Reserved.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
