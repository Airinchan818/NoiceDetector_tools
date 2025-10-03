# NoiceDetector_tools
A Vision Transformer–based AI model for detecting soft noise in video and image streams.
NDM is inspired by Diffusion Models and trained with Gaussian Noise distribution to identify even the smallest interference that is often invisible to the human eye 👀.

🚀 Features:
----------------------------
- 🎥 Realtime & Recorded Video Detection
- 📷 Flexible choice of internal or external camera for real-time detection
- 🔄 Video resize support while detection is running

🤔 Why NDM?
-----------------------
While big noise (like heavy blur or distortion) can be seen easily, soft noise is harder to detect.
NDM is designed to:
- ✅ Detect subtle interference that humans often miss
- 🛡️ Identify potential sabotage, camera interference, or hidden errors
- ⚡ Provide an extra layer of video security

💻 Minimum Hardware Requirements:
----------------------------------------
- 🖥️ RAM: 8 GB
- ⚙️ OS: Any OS that supports Python & PyTorch
- 🔲 CPU: Intel i3 (7th Gen) or higher

🧠 Training Process

NDM was trained using Self-Supervised Learning:
-----------------------------------------------------
- 📊 Dataset: 90k real images
- ➕ Generated noisy images: ~180k samples
- 🎯 Noise labels were produced using Gaussian Noise makers (similar to Diffusion Model training)
- ⚙️ Frameworks: PyTorch + OpenCV

📦 Installation:
--------------------
bash 
```
  git clone https://github.com/Airinchan818/NoiceDetector_tools.git
  cd NoiceDetector_tools
```
tools need:
--------------
- Torch
- OpenCv
- PyQt5

# ☕ Support me in creating more innovative AI projects: https://lnkd.in/dwE7gSdF

credit :
-------------------

- Researcher, Engineers , and programmers Project : Candra Alpin Gunawan 
- datasets: AI vs. Human-Generated Images  by kaggle users (Alessandra Sala (Owner),Manuela Jeyaraj (Editor),Toma Ijatomi (Editor),Margarita Pitsiani (Editor)) with Apache 2.0 license 
    datasets link: https://lnkd.in/eaGzCSt5
- image processing: opencv 
    link: https://lnkd.in/e3BC7JZj
- model trainer and builder: torch 
    link: https://pytorch.org/
- GUI Makers: PyQt5 
  link: https://lnkd.in/etAS_VjF.


Contact Email: hinamatsuriairin@gmail.com
