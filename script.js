document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('camera');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');
    const searchCaptureButton = document.getElementById('search-capture');
    const ctx = canvas.getContext('2d');
    let capturedImage = null;
    let processing = false;
    const registrationForm = document.getElementById('registrationForm');
    const votingForm = document.getElementById('votingForm');
    const votingCard = document.getElementById('votingCard');
    const verificationCard = document.getElementById('verificationCard');
    const verifyButton = document.getElementById('verifyButton');
    let voteSignature = null;

    // Initialize camera with lower resolution
    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                } 
            });
            video.srcObject = stream;
        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Could not access camera. Please ensure you have granted camera permissions.');
        }
    }

    // Optimize image before upload
    function optimizeImage(imageData, maxSize = 1280) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;

                if (width > height && width > maxSize) {
                    height = Math.round((height * maxSize) / width);
                    width = maxSize;
                } else if (height > maxSize) {
                    width = Math.round((width * maxSize) / height);
                    height = maxSize;
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);
                resolve(canvas.toDataURL('image/jpeg', 0.8));
            };
            img.src = imageData;
        });
    }

    // Capture image from camera
    captureButton.addEventListener('click', async function() {
        if (processing) return;
        processing = true;
        
        try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            capturedImage = await optimizeImage(canvas.toDataURL('image/jpeg'));
            searchCaptureButton.disabled = false;
        } catch (error) {
            console.error('Error capturing image:', error);
            alert('Error capturing image');
        } finally {
            processing = false;
        }
    });

    // Search captured image
    searchCaptureButton.addEventListener('click', async function() {
        if (!capturedImage || processing) return;
        processing = true;
        
        try {
            const response = await fetch(capturedImage);
            const blob = await response.blob();
            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');
            
            const result = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await result.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing image');
        } finally {
            processing = false;
        }
    });

    // Handle file upload
    const fileInput = document.getElementById('image');
    fileInput.addEventListener('change', async function(e) {
        if (processing) return;
        processing = true;
        
        const file = e.target.files[0];
        if (file) {
            try {
                const formData = new FormData();
                formData.append('image', file);

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image');
            }
        }
        processing = false;
    });

    // Display results with loading state
    function displayResults(data) {
        const resultsDiv = document.getElementById('results');
        if (data.error) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }

        let html = '<div class="results-container">';
        
        if (data.processing_time) {
            html += `<div class="text-muted mb-3">Processing time: ${data.processing_time}s</div>`;
        }
        
        if (data.text) {
            html += `<div class="mb-3">
                <h6>Detected Text:</h6>
                <p>${data.text}</p>
            </div>`;
        }

        if (data.objects && data.objects.length > 0) {
            html += `<div class="mb-3">
                <h6>Detected Objects:</h6>
                <ul class="list-group">`;
            data.objects.forEach(obj => {
                html += `<li class="list-group-item">
                    ${obj.class} (${(obj.confidence * 100).toFixed(1)}%)
                </li>`;
            });
            html += '</ul></div>';
        }

        if (data.quality) {
            html += `<div class="mb-3">
                <h6>Image Quality:</h6>
                <p>${data.quality}</p>
            </div>`;
        }

        html += '</div>';
        resultsDiv.innerHTML = html;
    }

    // Handle registration
    registrationForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            name: document.getElementById('name').value,
            id_number: document.getElementById('idNumber').value,
            email: document.getElementById('email').value
        };

        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            
            if (response.ok) {
                alert('Registration successful! You can now cast your vote.');
                votingCard.style.display = 'block';
                registrationForm.style.display = 'none';
            } else {
                alert(data.error || 'Registration failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Registration failed. Please try again.');
        }
    });

    // Handle voting
    votingForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const selectedCandidate = document.querySelector('input[name="candidate"]:checked');
        if (!selectedCandidate) {
            alert('Please select a candidate');
            return;
        }

        try {
            const response = await fetch('/vote', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    candidate_id: selectedCandidate.value
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                voteSignature = data.signature;
                document.getElementById('voteSignature').textContent = voteSignature;
                verificationCard.style.display = 'block';
                votingCard.style.display = 'none';
                alert('Vote cast successfully!');
                updateResults();
            } else {
                alert(data.error || 'Voting failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Voting failed. Please try again.');
        }
    });

    // Handle vote verification
    verifyButton.addEventListener('click', async function() {
        if (!voteSignature) {
            alert('No vote to verify');
            return;
        }

        try {
            const response = await fetch('/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    signature: voteSignature
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                alert(data.verified ? 'Vote verified successfully!' : 'Vote verification failed');
            } else {
                alert(data.error || 'Verification failed');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Verification failed. Please try again.');
        }
    });

    // Update results periodically
    async function updateResults() {
        try {
            const response = await fetch('/results');
            const results = await response.json();
            
            const resultsDiv = document.getElementById('results');
            if (Object.keys(results).length === 0) {
                resultsDiv.innerHTML = '<div class="text-center text-muted"><p>No votes cast yet</p></div>';
                return;
            }

            let html = '<div class="list-group">';
            for (const [candidate, votes] of Object.entries(results)) {
                html += `
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        ${candidate}
                        <span class="badge bg-primary rounded-pill">${votes}</span>
                    </div>
                `;
            }
            html += '</div>';
            resultsDiv.innerHTML = html;
        } catch (error) {
            console.error('Error updating results:', error);
        }
    }

    // Update results every 5 seconds
    setInterval(updateResults, 5000);
    updateResults(); // Initial update

    // Initialize camera when page loads
    initCamera();
}); 