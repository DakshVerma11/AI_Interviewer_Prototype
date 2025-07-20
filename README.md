
# AI-Powered Virtual Interviewer Platform

> **Revolutionizing Student Interview Preparation & Company Assessment**

## Links

* **[Presentation](https://docs.google.com/presentation/d/16DFZeK78ABYQ3xu92gWPgLPZd-nJa563J7Zev1EX4-4/edit?usp=sharing)**
* **[Demo Video](https://www.youtube.com/watch?v=CAdY-7bwW7E)**

---

## Overview

The AI-Powered Virtual Interviewer Platform is an innovative web application designed to help students practice interviews in a realistic, data-driven environment and assist companies in efficiently assessing candidates at scale. This platform leverages AI and computer vision to simulate real interview scenarios, record and analyze video/audio responses, detect cheating via eye tracking, and provide actionable feedback.

---

## Features

### For Students

* **Login/Register:** Secure authentication and personalized user dashboard.
* **Resume Upload & Role Selection:** Upload your resume and specify the role you are targeting. The AI tailors interview questions accordingly.
* **Start Interview:** Receive role-specific questions, record your responses via video (with audio), and progress through an asynchronous interview session.
* **Eye Tracking Detection:** Real-time computer vision analysis to detect off-camera glances and suspicious behavior for exam integrity.
* **Results & Feedback:** Review your interview analytics, including eye tracking results and completion stats. (Full feedback and scoring coming soon.)
* **History:** View previous interview sessions and track your progress over time.

### For Companies / Admin

* **Scalable Assessment:** Efficiently shortlist and assess large candidate pools.
* **Cheating Detection:** Automated flagging of integrity issues.
* **Aggregate Analytics:** (Planned) Cohort and candidate-level reporting.

---

## Technology Stack

* **Frontend:** HTML5, CSS3, JavaScript (vanilla, no framework), Responsive Design
* **Backend:** Python 3, Flask (REST API), CSV/JSON-based storage
* **Media Capture:** WebRTC / HTML5 getUserMedia for video/audio streaming
* **Computer Vision:** OpenCV (Haar Cascades) for face and eye tracking
* **Async Processing:** Python threading for video analysis
* **Data:** File-based storage per user (scalable to database in production)
* **Security:** SHA256 password hashing, session cookies

---

## How It Works

1. **Register/Login:** Create an account or log in.
2. **Profile Setup:** Upload your resume (PDF) and specify your target job role.
3. **Interview Prep:** Once your interview is ready, start the session to receive tailored questions.
4. **Interview Recording:** Answer each question on video (audio recorded separately), at 24 FPS for optimal performance.
5. **Eye Tracking:** The system analyzes your video for eye/face movement and flags potential "cheating."
6. **Results:** After processing, view your interview analytics and stats on the dashboard.

---

## Quick Start (Development)

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DakshVerma11/CodeClash2.0_Prototype.git
   cd CodeClash2.0_Prototype
   ```

2. **Install Python Dependencies**

   ```bash
   pip install flask opencv-python numpy
   ```

3. **Run the Application**

   ```bash
   python app.py
   ```

   > The server will start on `http://localhost:5000`

4. **Open in Browser**

   * Go to `http://localhost:5000`
   * Register a new account and start practicing!

---

## Demo

* **[Presentation](https://docs.google.com/presentation/d/16DFZeK78ABYQ3xu92gWPgLPZd-nJa563J7Zev1EX4-4/edit?usp=sharing)**
* **[Demo Video](https://www.youtube.com/watch?v=CAdY-7bwW7E)**


---

## Roadmap

* [x] Secure login/register and user dashboard
* [x] Resume upload and role selection
* [x] Optimized 24 FPS video/audio recording
* [x] Real-time eye tracking and cheating detection
* [x] Results page with basic analytics
* [ ] AI-driven question generation (coming soon)
* [ ] Speech analysis and scoring (coming soon)
* [ ] Full feedback/coaching reports (coming soon)
* [ ] Admin/company dashboards and bulk analytics

---

## Contributing

We welcome contributions! Please create issues or pull requests for feature requests, bug reports, or improvements.

---

## License

This project is released under the MIT License.

---

## Contact

* **Project Lead:** [Daksh Verma](https://github.com/DakshVerma11)

---

*Empowering every student to ace their interviews and every recruiter to hire with confidence.*
