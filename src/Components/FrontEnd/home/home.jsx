import React from "react";
import "./home.css";
import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();
  return (
    <div className="home">
      {/* Header Section */}
      <header className="home-header">
        <h1>Cancer Detection System</h1>
        <p>AI-Powered Solution for Early Cancer Detection</p>
        <button
          className="cta-button"
          onClick={() => navigate("/upload")}
        >
          Get Started
        </button>
      </header>

      {/* Features Section */}
      <section className="features">
        <h2>Why Choose Our System?</h2>
        <div className="features-container">
          <div className="feature-card">
            <i className="feature-icon fas fa-search"></i>
            <h3>Accurate Detection</h3>
            <p>Using advanced AI to provide precise results.</p>
          </div>
          <div className="feature-card">
            <i className="feature-icon fas fa-lock"></i>
            <h3>Privacy First</h3>
            <p>Your images are processed securely and confidentially.</p>
          </div>
          <div className="feature-card">
            <i className="feature-icon fas fa-lightbulb"></i>
            <h3>Explainable AI</h3>
            <p>Understand the reasoning behind every diagnosis.</p>
          </div>
        </div>
      </section>

      {/* Footer Section */}
      {/* <footer className="home-footer">
        <p>&copy; 2025 Cancer Detection System. All rights reserved.</p>
      </footer> */}
    </div>
  );
}

export default Home;
