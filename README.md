# Smooth Operator - AI Call Assistant

Terminates spam calls? Operates smoothly? No, it's a **Smooth Operator** - an educational project demonstrating how to build an AI-powered personal receptionist that handles your phone calls.

This repository complements a blog series where we build a Voice AI agent from scratch, covering everything from handling real-time audio streams to generating human-like voice responses with LLMs.

## 📚 Blog Series Roadmap

This project is divided into chapters, each corresponding to a lesson in the blog series.

| Chapter | Status | Topic | Description |
| :--- | :--- | :--- | :--- |
| [**Chapter 1**](https://www.magd.dev/blog/ai-call-assistant-chapter-1-building-the-listening-layer) | ✅ Done | **Hearing Ears** | Accepting inbound calls and performing real-time Speech-to-Text (STT) transcription. |
| **Chapter 2** | ✅ Done | **The Brain** | Understanding context with LLMs, generating answers, and managing conversation history. |
| **Chapter 3** | 🚧 Planned | **The Voice** | Speaking back to the caller using low-latency Text-to-Speech (TTS). |

---

## 🚀 Getting Started

Follow these instructions to set up the project for **Chapter 1**.

### Prerequisites

*   **Python 3.12+**
*   **[uv](https://docs.astral.sh/uv/)** (recommended for dependency management) or pip
*   **[Twilio Account](https://www.twilio.com/)** (with a phone number)
*   **[Ngrok](https://ngrok.com/)** (to expose your local server)
*   **[ElevenLabs API Key](https://elevenlabs.io/)** or **[Mistral API Key](https://mistral.ai/)** (for STT) 

### 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smooth-operator.git
    cd smooth-operator
    ```

2.  **Install dependencies:**
    Using `uv`:
    ```bash
    uv sync
    ```


3.  **Environment Setup:**
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    
    Edit `.env` and fill in your keys:
    ```ini
    ELEVENLABS_API_KEY=sk_...
    PUBLIC_BASE_URL=https://your-ngrok-url.ngrok-free.app
    ```

### ▶️ Running the Application

1.  **Start the Server:**
    ```bash
    uv run uvicorn app.main:app --reload
    ```


2.  **Expose your local server:**
    Run ngrok to tunnel traffic to port 8000:
    ```bash
    ngrok http 8000
    ```
    Copy the forwarding URL (e.g., `https://abcdef.ngrok-free.app`) and update `PUBLIC_BASE_URL` in your `.env`.

3.  **Configure Twilio:**
    *   Go to your Twilio Console > Phone Numbers > Manage > Active Numbers.
    *   Click on your number.
    *   Under **Voice & Fax** > **A Call Comes In**:
        *   Select **Webhook**.
        *   URL: `{YOUR_NGROK_URL}/twilio/inbound` (e.g., `https://abcdef.ngrok-free.app/twilio/inbound`)
        *   HTTP Method: **POST**
    *   Save configuration.

### 🧪 Testing (Chapter 1)

1.  Ensure your server is running and the Twilio webhook is configured.
2.  Call your Twilio phone number.
3.  Speak into the phone.
4.  Watch your server terminal—you should see real-time transcripts appearing as you speak!
    ```text
    [CA12345...] Twilio stream started
    [CA12345...] PARTIAL: Hello this is
    [CA12345...] COMMITTED: Hello, this is a test call.
    ```

## 📂 Project Structure

*   `app/main.py`: Entry point for the FastAPI application.
*   `app/api/routes/twilio.py`: Handles Twilio webhooks and WebSocket upgrades.
*   `app/services/twilio/stream.py`: Manages the WebSocket media stream and audio buffering.
*   `app/services/stt/`: Integrates with selected STT provider.
