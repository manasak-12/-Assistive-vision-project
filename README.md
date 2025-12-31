<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Based Assistive Vision System for the Visually Impaired</title>
</head>
<body>

<h1>AI-Based Assistive Vision System for the Visually Impaired</h1>

<h2>Overview</h2>
<p>
The <strong>AI-Based Assistive Vision System for the Visually Impaired</strong> is a fully offline, real-time assistive application
designed to help visually impaired users understand their surroundings through spoken feedback.
The system captures live video, analyzes the environment using artificial intelligence and computer vision,
and converts meaningful visual information into natural audio descriptions.
</p>

<p>
The system operates entirely offline, ensuring low latency, reliability, and strong privacy protection.
It runs on standard CPU-based hardware without requiring cloud services or GPUs.
</p>

<hr>

<h2>Key Features</h2>
<ul>
    <li>Real-time object detection and identification</li>
    <li>Face detection and recognition of registered individuals</li>
    <li>Facial emotion detection for basic social context awareness</li>
    <li>Optical Character Recognition (OCR) to read visible text</li>
    <li>Offline speech-to-text for voice-based commands</li>
    <li>Offline text-to-speech for natural spoken feedback</li>
    <li>Automatic periodic scene narration</li>
    <li>Web-based dashboard for monitoring and face management</li>
</ul>

<hr>

<h2>System Architecture</h2>
<p>
The system follows a modular, multithreaded architecture to ensure non-blocking real-time performance.
Different components operate independently while sharing data through controlled memory access.
</p>

<ul>
    <li><strong>Vision Processing Thread:</strong> Handles live camera input and object detection</li>
    <li><strong>Audio Processing Thread:</strong> Listens for offline voice commands</li>
    <li><strong>Background Worker Pool:</strong> Executes OCR, face recognition, and emotion analysis</li>
    <li><strong>Text-to-Speech Thread:</strong> Delivers spoken responses without interrupting vision</li>
    <li><strong>Flask Backend:</strong> Coordinates system modules and provides a web dashboard</li>
</ul>

<hr>

<h2>Technologies Used</h2>
<ul>
    <li><strong>Programming Language:</strong> Python</li>
    <li><strong>Computer Vision:</strong> OpenCV</li>
    <li><strong>Object Detection:</strong> YOLO (Ultralytics)</li>
    <li><strong>Speech Recognition:</strong> Vosk (Offline)</li>
    <li><strong>Text-to-Speech:</strong> pyttsx3 (Offline)</li>
    <li><strong>OCR:</strong> Tesseract / EasyOCR</li>
    <li><strong>Web Framework:</strong> Flask, Flask-SocketIO</li>
    <li><strong>Data Handling:</strong> NumPy, Pandas</li>
    <li><strong>Version Control:</strong> Git, GitHub</li>
</ul>

<hr>

<h2>Functional Capabilities</h2>
<ul>
    <li>Continuous live video analysis</li>
    <li>Voice-controlled interaction using natural commands</li>
    <li>Spoken descriptions of objects, people, emotions, and text</li>
    <li>Dynamic face registration without restarting the system</li>
    <li>Fully offline execution for privacy and reliability</li>
</ul>

<hr>

<h2>Hardware Requirements</h2>
<ul>
    <li>Laptop or desktop with a multi-core CPU</li>
    <li>Minimum 8 GB RAM (recommended)</li>
    <li>Webcam (built-in or external)</li>
    <li>Microphone and speakers or headphones</li>
    <li>No GPU or specialized hardware required</li>
</ul>

<hr>

<h2>How to Run</h2>
<ol>
    <li>Clone the repository:
        <pre>git clone https://github.com/your-username/your-repository-name.git</pre>
    </li>
    <li>Create and activate a virtual environment</li>
    <li>Install dependencies:
        <pre>pip install -r requirements.txt</pre>
    </li>
    <li>Run the application:
        <pre>python app.py</pre>
    </li>
    <li>Open the web dashboard in a browser:
        <pre>http://localhost:5000</pre>
    </li>
</ol>

<hr>

<h2>Use Cases</h2>
<ul>
    <li>Environmental awareness for visually impaired users</li>
    <li>Reading printed text such as labels and documents</li>
    <li>Identifying familiar individuals</li>
    <li>Understanding basic social and emotional context</li>
</ul>

<hr>

<h2>Limitations</h2>
<ul>
    <li>Reduced accuracy in very low-light conditions</li>
    <li>Face recognition performance drops for extreme angles or occlusions</li>
    <li>Currently supports English language only</li>
    <li>Optimized for desktop and laptop environments</li>
</ul>

<hr>

<h2>Future Enhancements</h2>
<ul>
    <li>Multilingual speech and OCR support</li>
    <li>Mobile and embedded device deployment</li>
    <li>Improved spatial awareness and depth estimation</li>
    <li>Advanced emotion and activity recognition</li>
</ul>

<hr>

<h2>Author</h2>
<p>
<strong>K Manasa</strong><br>
B.E. Computer Science and Engineering (Data Science)<br>
SJB Institute of Technology, VTU
</p>

</body>
</html>
