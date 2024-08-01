import React, { useState, useEffect } from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';

import Prediction from './Prediction';
import Dashboard from './Visualization';

function App() {
  const [comments, setComments] = useState([]);

  useEffect(() => {
    const fetchComments = async () => {
      try {
        const response = await fetch('./data-process.json');
        const data = await response.json();
        setComments(data);
      } catch (error) {
        console.error('Error fetching comments:', error);
      }
    };
    fetchComments();
  }, []);

  return (
    <Router>
      <div className="App">
        <nav>
          <ul>
            <li>
              <Link to="/">Prediction</Link>
            </li>
            <li>
              <Link to="/visualization">Visualization</Link>
            </li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<Prediction />} />
          <Route path="/visualization" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
