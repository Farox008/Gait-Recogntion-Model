# 👋 Hello! Welcome to our Gait Recognition Model

This project is the result of a lot of hard work by me and my team. We wanted to build something that doesn't just "see" people, but really understands how they move. Gait recognition—identifying someone just by their walk—is a tough challenge, but it’s incredibly powerful for security and smart monitoring.

We’ve combined two different ways of looking at movement: **the way your joints move (Skeletons)** and **the shape of your body as you walk (Silhouettes)**. By putting these two together, we created a "fingerprint" for your walk that’s hard to fool!

---

## ✨ What makes our project different?

*   **We look at everything**: Instead of just using one method, our model uses YOLOv8-Pose and YOLOv8-Seg together. This makes it super robust, even if the lighting is poor or someone is wearing a big coat.
*   **Built for different angles**: One of the biggest problems in gait recognition is that people look different from the side than from the front. We solved this with our **Camera-Specific Smart Gallery**. You can train the AI to recognize people's walk from a specific camera's perspective, making it much more accurate.
*   **Super Fast**: We used FAISS (Facebook AI Similarity Search) to make sure the identification happens instantly. It’s built to handle live streams in real-time.

---


## Our Team 

Without my team this project would not have been possible. I have to specially thank Angelina K Joseph for her creative ideas and her hard work in making this project a success,Belda Ben Thomas for doing documentation and paper subbmisions and especially Allena Varghese for collecting the dataset and her hard work in making this project a success.

## 🧩 How it works 

Under the hood, our pipeline does a few clever things:
1.  **Tracking**: It spots a person and follows them through the video.
2.  **The Details**: It draws an outline of their silhouette and maps out where their joints are.
3.  **The Hybrid Brain**: We use "3D Hybrid Blocks." This is a fancy way of saying the model captures the **rhythm** of the walk over time, not just a still picture.
4.  **Body Mapping**: It splits the person into 16 parts from head to toe, so it can pay separate attention to how feet move versus how arms swing.

---

## 🏃 Getting Started (Your First Run)

### 1. Prerequisites
*   Make sure you have Python 3.10+.
*   A GPU makes things much faster (CUDA), but CPU works too.
*   **Important**: Install **Git LFS**! Our AI's "brain" files are large, so they need LFS to download properly.

### 2. Quick Setup
```bash
git clone https://github.com/Farox008/Gait-Recogntion-Model.git
cd Gait-Recogntion-Model

# Download the model brains
git lfs pull

# Install the bits and pieces
pip install -r requirements.txt
```

### 3. Start the UI!
Just run this and follow the menu:
```bash
python run.py
```

---

## 📹 Pro-Tip: Use Camera IDs
If you're using this with multiple cameras, use the **Camera ID** feature (like `Gate-A` or `Lobby`). This tells the AI to create a specific "memory bank" for that angle, which gets you much closer to perfect accuracy.

---

## 📊 Performance (Our Team's Results)

We tested this model on a **custom dataset that me and my team personally collected**, and we’re really proud of the results:

*   **93.0% Overall Accuracy**: Out of 43 tests, 40 were spot on!
*   **Why it's so high**: Because we used high-quality segmentation and multiple training clips per person.
*   **Room for improvement**: The few misses happened when someone was walking directly toward the camera—a view where it's hard to see the leg swing. Using the **Camera ID** feature usually fixes this!

---

## 📦 What are these files?
- `deepgaitv2.pt`: The silhouette expert.
- `skeletongait++.pt`: The joint movement expert.
- `fusion.pt`: The master logic that joins them together.
- `yolov8n*.pt`: The vision models that spot the people in the first place.

---

## 🚀 Recent Updates
- ✅ **One-Script-To-Rule-Them-All**: The new `run.py` makes everything easy.
- ✅ **Custom Folders**: Point the model at any video or folder.
- ✅ **Pixel-Perfect Silhouettes**: Switched to YOLOv8-Seg for sharp, clean outlines.
- ✅ **Viewpoint Memory**: Your trained people are now saved per-camera.
