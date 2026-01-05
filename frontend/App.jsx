const { useState, useEffect, useRef } = React;

function App() {
  const [theme, setTheme] = useState("dark");
  const [view, setView] = useState("home"); 
  const [result, setResult] = useState(null);
  
  // State for Combined Uploads
  const [combinedSpiral, setCombinedSpiral] = useState(null);
  const [combinedVoice, setCombinedVoice] = useState(null);

  const spiralInputRef = useRef(null);
  const voiceInputRef = useRef(null);
  const combinedSpiralRef = useRef(null);
  const combinedVoiceRef = useRef(null);

  // ✅ URL DEBUGGER: This lets you type in the URL to see pages
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const status = params.get("status");

    if (status === "detected") {
      setView("result");
      setResult(true);
    } else if (status === "healthy") {
      setView("result");
      setResult(false);
    }
  }, []);

  // Apply Theme & Result Backgrounds
  useEffect(() => {
    document.body.classList.remove("light", "dark", "bg-red", "bg-green");
    document.body.classList.add(theme);

    if (view === "result") {
      if (result === true) document.body.classList.add("bg-red");
      else if (result === false) document.body.classList.add("bg-green");
    }
  }, [theme, view, result]);

  // --- HANDLER: Single File Upload ---
  const handleSingleUpload = async (event, endpoint) => {
    const file = event.target.files[0];
    if (!file) return;

    setView("loading");
    const formData = new FormData();
    formData.append("file", file);

    await sendToBackend(endpoint, formData);
  };

  // --- HANDLER: Combined Upload ---
  const handleCombinedSubmit = async () => {
    if (!combinedSpiral || !combinedVoice) {
      alert("Please upload both files first!");
      return;
    }

    setView("loading");
    const formData = new FormData();
    formData.append("spiral_file", combinedSpiral);
    formData.append("voice_file", combinedVoice);

    await sendToBackend("/predict-combined", formData);
  };

  // --- API HELPER (UPDATED) ---
  const sendToBackend = async (endpoint, formData) => {
    try {
      // ✅ CONNECTED TO LIVE RENDER BACKEND
      const response = await fetch(`https://parkinsons-backend-40ys.onrender.com${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data.detected);
      setView("result");
    } catch (error) {
      console.error("Error:", error);
      alert("Backend connection failed. Check console.");
      setView("home");
    }
  };

  const resetTest = () => {
    // Clear the URL so we can go back home cleanly
    window.history.pushState({}, document.title, window.location.pathname);
    setView("home");
    setResult(null);
    setCombinedSpiral(null);
    setCombinedVoice(null);
  };

  return (
    <div>
      {/* BACKGROUND SVG */}
      <svg className="bg-svg" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="hookGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="50%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
        </defs>
        <path d="M 200 100 C 700 0 900 300 600 500 C 350 650 400 900 800 1000" stroke="url(#hookGrad)" strokeWidth="260" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>

      <div className="bg-overlay"></div>
      <div className="bg-accent"></div>
      <div className="bg-design"></div>
      <div className="bg-grid"></div>
      <div className="bg-spotlight"></div>

      <header>
        <h2>🧠 Parkinson’s Detection System</h2>
        <div className="toggle" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "Light Mode" : "Dark Mode"}
        </div>
      </header>

      <div className="main">
        
        {/* VIEW: HOME */}
        {view === "home" && (
          <div className="cards">
            
            {/* 1. Spiral Card */}
            <div className="card">
              <h2>✍️ Spiral Detection</h2>
              <p>Upload a spiral drawing</p>
              <input 
                type="file" 
                ref={spiralInputRef} 
                style={{display: 'none'}} 
                onChange={(e) => handleSingleUpload(e, "/predict-spiral")} 
                accept="image/*"
              />
              <button className="button" onClick={() => spiralInputRef.current.click()}>
                Upload Spiral Image
              </button>
            </div>

            {/* 2. Voice Card */}
            <div className="card">
              <h2>🎤 Voice Detection</h2>
              <p>Upload a .wav audio file</p>
              <input 
                type="file" 
                ref={voiceInputRef} 
                style={{display: 'none'}} 
                onChange={(e) => handleSingleUpload(e, "/predict-voice")} 
                accept=".wav"
              />
              <button className="button" onClick={() => voiceInputRef.current.click()}>
                Upload Audio (.wav)
              </button>
            </div>

            {/* 3. Combined Card */}
            <div className="card best">
              <h2>🔀 Combined Detection</h2>
              <p>Upload both files below:</p>
              
              {/* Box 1: Spiral */}
              <input 
                type="file" 
                ref={combinedSpiralRef} 
                style={{display: 'none'}} 
                accept="image/*"
                onChange={(e) => setCombinedSpiral(e.target.files[0])} 
              />
              <button 
                className="button" 
                style={{marginBottom: '10px'}}
                onClick={() => combinedSpiralRef.current.click()}
              >
                {combinedSpiral ? "✓ Spiral Uploaded" : "1. Upload Spiral"}
              </button>

              {/* Box 2: Voice */}
              <input 
                type="file" 
                ref={combinedVoiceRef} 
                style={{display: 'none'}} 
                accept=".wav"
                onChange={(e) => setCombinedVoice(e.target.files[0])} 
              />
              <button 
                className="button" 
                style={{marginBottom: '15px'}}
                onClick={() => combinedVoiceRef.current.click()}
              >
                {combinedVoice ? "✓ Voice Uploaded" : "2. Upload Voice (.wav)"}
              </button>

              {/* Submit Button */}
              {combinedSpiral && combinedVoice && (
                <button className="button" onClick={handleCombinedSubmit} style={{background: 'var(--primary)'}}>
                  Run Combined Test
                </button>
              )}
            </div>
          </div>
        )}

        {/* VIEW: LOADING */}
        {view === "loading" && (
          <div className="card" style={{textAlign: "center"}}>
            <h2>⏳ Analyzing...</h2>
            <p>Our AI is processing your file(s).</p>
          </div>
        )}

        {/* VIEW: RESULT */}
        {view === "result" && (
          <div className="card" style={{textAlign: "center", border: "4px solid white"}}>
            <h1 style={{fontSize: "3rem", margin: "0"}}>
              {result ? "⚠️ DETECTED" : "✅ HEALTHY"}
            </h1>
            <p style={{fontSize: "1.2rem", marginTop: "10px"}}>
              {result 
                ? "Signs of Parkinson's were found in the analysis." 
                : "No signs of Parkinson's were detected."}
            </p>
            <button className="button" onClick={resetTest} style={{marginTop: "20px", background: "white", color: "black"}}>
              Start New Test
            </button>
          </div>
        )}

      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);