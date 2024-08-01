import React, { useState,useEffect } from 'react';
import './App.css';

const Prediction = () => {
  const [text, setText] = useState('');
  // const [results, setResults] = useState({ absa: {}, sa: '' });
  const [results, setResults] = useState({ absa1: {}, absa2: {}, absa3: {}, sa: '' });

  const modelNames = {
    absa1: 'MultiTask ',
    absa2: 'MultiBranch ',
    absa3: 'Single classifier '
  };
  const handlePredict = async () => {
    const fetchOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    };
    try {
      const absa1Response = await fetch(`http://localhost:5000/predict/absa1`, fetchOptions);
      if (!absa1Response.ok) throw new Error('Failed to fetch ABSA1 data');
      const absa1Data = await absa1Response.json();

      const absa2Response = await fetch(`http://localhost:5000/predict/absa2`, fetchOptions);
      if (!absa2Response.ok) throw new Error('Failed to fetch ABSA2 data');
      const absa2Data = await absa2Response.json();

      const absa3Response = await fetch(`http://localhost:5000/predict/absa3`, fetchOptions);
      if (!absa3Response.ok) throw new Error('Failed to fetch ABSA3 data');
      const absa3Data = await absa3Response.json();
    
      const saResponse = await fetch(`http://localhost:5000/predict/sa`, fetchOptions);
      if (!saResponse.ok) throw new Error('Failed to fetch SA data');
      const saData = await saResponse.json();
    
      // setResults({ absa: absaData, sa: saData.sentiment });
      setResults({ absa1: absa1Data, absa2: absa2Data, absa3: absa3Data, sa: saData.sentiment });

    } catch (error) {
      console.error('Prediction error:', error);
    }

  };
  return (
    <div className="App">
      <h1>Welcome to my sentiment analysis demo</h1>
      <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter your input text"/>

      <div>
        <button onClick={handlePredict}>Predict</button>
      </div>

      <div className="result">

        <div style={{display: "flex", alignItems: "center"}}>
          <h2>Overall sentiment:</h2>
          {results.sa ? 
            <span style={{marginLeft: "10px", fontWeight: "bold"}}>{results.sa}</span> : 
            <span style={{fontWeight: "normal", marginLeft: "10px"}}>No overall sentiment to display</span>
          }
        </div>

        {/* <div style={{display: "flex", flexDirection: "column", alignItems: "flex-start"}}>
          <div style={{display: "flex", alignItems: "center"}}>
            <h2>Aspect based Sentiment Results:</h2>
            {Object.keys(results.absa).length === 0 && (
              <span style={{fontWeight: "normal", marginLeft: "10px"}}>No relevant aspects to display</span>
            )}
          </div>
          {Object.keys(results.absa).length > 0 && (
            <div style={{marginLeft: "10px"}}>
              {Object.entries(results.absa).filter(([_, sentiment]) => sentiment !== 'None').map(([aspect, sentiment], index) => (
                <p key={index} style={{marginBottom: "10px"}}>{aspect.toUpperCase()}: {sentiment}</p>
              ))}
            </div>
          )}
        </div> */}
        {/* {['absa1', 'absa2', 'absa3'].map((model) => (
          <div style={{display: "flex", flexDirection: "column", alignItems: "flex-start"}} key={model}>
            <div style={{display: "flex", alignItems: "center"}}>
              <h2>{model.toUpperCase()} Sentiment Results:</h2>
              {Object.keys(results[model]).length === 0 && (
                <span style={{fontWeight: "normal", marginLeft: "10px"}}>No relevant aspects to display</span>
              )}
            </div>
            {Object.keys(results[model]).length > 0 && (
              <div style={{marginLeft: "10px"}}>
                {Object.entries(results[model]).filter(([_, sentiment]) => sentiment !== 'None').map(([aspect, sentiment], index) => (
                  <p key={index} style={{marginBottom: "10px"}}>{aspect.toUpperCase()}: {sentiment}</p>
                ))}
              </div>
            )}
          </div>
        ))} */}


        {['absa1', 'absa3', 'absa2'].map((model) => (
          <div style={{display: "flex", flexDirection: "column", alignItems: "flex-start"}} key={model}>
            <div style={{display: "flex", alignItems: "center"}}>
              <h2>PhoBERT with { modelNames[model]}:</h2>
              {Object.keys(results[model]).length === 0 && (
                <span style={{fontWeight: "normal", marginLeft: "10px"}}>No relevant aspects to display</span>
              )}
            </div>
            {Object.keys(results[model]).length > 0 && (
              <div style={{marginLeft: "10px"}}>
                {Object.entries(results[model]).filter(([_, sentiment]) => sentiment !== 'None').map(([aspect, sentiment], index) => (
                  <p key={index} style={{marginBottom: "10px"}}>{aspect.toUpperCase()}: {sentiment}</p>
                ))}
              </div>
            )}
          </div>
        ))}





      </div>
    </div>
  );
}

export default Prediction ;
