import React, { useState, useCallback } from 'react';

// --- Configuration ---
const BACKEND_URL = 'https://brain-tumour-api.onrender.com/predict_batch';

// --- Helper Components ---

const UploadIcon = () => (
    <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
    </svg>
);

const Loader = () => (
    <div className="loader"></div>
);

// --- Main Application Component ---

function App() {
    // --- State Management ---
    const [appStatus, setAppStatus] = useState('idle'); // 'idle', 'preview', 'loading', 'results'
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [analysisResults, setAnalysisResults] = useState([]);
    const [error, setError] = useState(null);

    // --- File Handling Logic ---

    const handleFiles = useCallback((files) => {
        const newFiles = Array.from(files).filter(file => 
            file.type.startsWith('image/') && !uploadedFiles.some(f => f.name === file.name)
        );

        if (newFiles.length > 0) {
            setUploadedFiles(prevFiles => [...prevFiles, ...newFiles]);
            setAppStatus('preview');
        }
    }, [uploadedFiles]);

    const removeFile = (indexToRemove) => {
        setUploadedFiles(prevFiles => {
            const newFiles = prevFiles.filter((_, index) => index !== indexToRemove);
            if (newFiles.length === 0) {
                setAppStatus('idle');
            }
            return newFiles;
        });
    };
    
    const resetState = () => {
        setUploadedFiles([]);
        setAnalysisResults([]);
        setError(null);
        setAppStatus('idle');
    };

    // --- Drag and Drop Handlers ---

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        e.currentTarget.classList.add('drag-over');
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    }, [handleFiles]);
    
    // --- API Call for Classification ---

    const handleClassification = async () => {
        if (uploadedFiles.length === 0) return;

        setAppStatus('loading');
        setError(null);

        const formData = new FormData();
        uploadedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Server returned an invalid response.' }));
                throw new Error(`Server error: ${response.statusText} - ${errorData.error || 'Unknown error'}`);
            }

            const data = await response.json();
            setAnalysisResults(data.results);
            setAppStatus('results');

        } catch (err) {
            console.error('Error during classification:', err);
            setError(err.message);
            setAppStatus('preview'); // Revert to preview on error
        }
    };

    // --- Render Logic ---

    return (
        <div className="app-container">
            <div className="container">
                <header className="app-header">
                    <h1 className="title">
                        Brain Tumor Analysis Platform
                    </h1>
                    <p className="subtitle">
                        An advanced tool for brain tumor classification and segmentation.
                    </p>
                </header>

                <main>
                    <div className="card">
                        
                        {appStatus === 'idle' && (
                            <div className="text-center">
                                <h2>Upload Scans</h2>
                                <div 
                                    className="drop-zone" 
                                    onDragOver={handleDragOver}
                                    onDragLeave={handleDragLeave}
                                    onDrop={handleDrop}
                                    onClick={() => document.getElementById('file-input')?.click()}
                                >
                                    <input type="file" id="file-input" className="hidden-input" accept="image/*" multiple onChange={(e) => handleFiles(e.target.files)} />
                                    <div className="drop-zone-content">
                                        <UploadIcon />
                                        <p><span>Click to upload</span> or drag and drop</p>
                                        <p className="drop-zone-subtitle">Batch processing enabled (PNG, JPG, etc.)</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {appStatus === 'preview' && (
                            <div className="preview-section">
                                <div className="preview-header">
                                    <h3>Uploaded Files ({uploadedFiles.length})</h3>
                                    <button onClick={resetState} className="btn btn-danger">Clear All</button>
                                </div>
                                {error && <div className="error-box" role="alert">{error}</div>}
                                <div className="preview-grid">
                                    {uploadedFiles.map((file, index) => (
                                        <div key={index} className="preview-item">
                                            <img src={URL.createObjectURL(file)} alt={file.name} />
                                            <div className="preview-item-overlay">
                                                <button onClick={() => removeFile(index)}>&times;</button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                <div className="text-center">
                                    <button onClick={handleClassification} className="btn btn-primary">
                                        Analyze Scans
                                    </button>
                                </div>
                            </div>
                        )}

                        {appStatus === 'loading' && (
                             <div className="loading-section">
                                <Loader />
                                <p>Analyzing {uploadedFiles.length} scan(s)...</p>
                            </div>
                        )}

                        {appStatus === 'results' && (
                            <div className="results-section">
                                <div className="results-header">
                                    <h2 className="title">Analysis Results</h2>
                                    <button onClick={resetState} className="btn btn-secondary">Analyze More</button>
                                </div>
                                <div className="results-grid">
                                    {analysisResults.map((result, index) => {
                                        const confidencePercentage = (result.confidence * 100).toFixed(2);
                                        return (
                                            <div key={index} className="result-card">
                                                <div className="result-images">
                                                    <h3 title={result.filename}>{result.filename}</h3>
                                                    <div className="image-pair">
                                                        <div>
                                                            <h4>Original Scan</h4>
                                                            <img src={result.original_image} alt="Original Scan" />
                                                        </div>
                                                        <div>
                                                            <h4>Tumor Segmentation</h4>
                                                            <img src={result.segmented_image} alt="Segmented Scan" />
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="result-details">
                                                    <h3>Analysis Details</h3>
                                                    <div className="details-content">
                                                        <p>Predicted Type: <span>{result.tumor_type}</span></p>
                                                        <div>
                                                            <p>Confidence:</p>
                                                            <div className="progress-bar-container">
                                                                <div className="progress-bar" style={{ width: `${confidencePercentage}%`}}></div>
                                                            </div>
                                                            <p className="confidence-percent">{confidencePercentage}%</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                </main>

                <footer className="app-footer">
                    <p>&copy; 2024 Brain Tumor Analysis Platform.</p>
                </footer>
            </div>
        </div>
    );
}

export default App;
