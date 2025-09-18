import os
from dotenv import load_dotenv
from shiny import App, render, ui, reactive
from huggingface_hub import InferenceClient
from datetime import datetime
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Initialize the Hugging Face AI client
def initialize_hf_client():
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("API key not found. Please set HF_TOKEN in your .env file or environment variables.")
    
    return InferenceClient(
        provider="auto",
        api_key=api_key,
    )

# Initialize the Google AI client for translation
def initialize_google_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    genai.configure(api_key=api_key)
    # Using a fast and efficient model for the translation task
    return genai.GenerativeModel('gemini-1.5-flash-latest')

# UI Definition
app_ui = ui.page_fluid(
    # Link to external CSS and JS files in the 'www' directory
    # CSS and JS are now incorporated directly into the app
    ui.tags.head(
        ui.tags.style("""
            /* Medical App Professional Styling */
            :root {
                --primary-color: #2563eb;
                --primary-dark: #1d4ed8;
                --secondary-color: #f8fafc;
                --accent-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --text-primary: #1f2937;
                --text-secondary: #6b7280;
                --text-muted: #9ca3af;
                --border-color: #e5e7eb;
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --radius-sm: 0.375rem;
                --radius-md: 0.5rem;
                --radius-lg: 0.75rem;
            }

            * {
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: #f4f7f9; /* Softer, more professional background */
                min-height: 100vh;
                margin: 0;
                padding: 0;
                color: var(--text-primary);
                line-height: 1.6;
            }

            .main-container {
                max-width: 1280px; /* Widen container for two-column layout */
                margin: 2rem auto;
                padding: 2rem;
                background: #ffffff; /* Cleaner solid white background */
                border-radius: var(--radius-lg);
                box-shadow: 0 10px 30px -15px rgba(0, 0, 0, 0.1); /* Softer, more subtle shadow */
            }

            .app-header {
                text-align: center;
                margin-bottom: 2rem;
                padding-bottom: 1.5rem;
                border-bottom: 2px solid var(--border-color);
            }

            .app-title {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--primary-color);
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }

            .app-subtitle {
                font-size: 1.2rem;
                color: var(--text-secondary);
                margin: 0.5rem 0 0 0;
                font-weight: 400;
            }

            .warning-box {
                background: linear-gradient(135deg, #fff3cd 0%, #fef3c7 100%);
                border: 1px solid #fbbf24;
                border-radius: var(--radius-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                color: #92400e;
                box-shadow: var(--shadow-sm);
            }

            .warning-box h5 {
                margin: 0 0 0.75rem 0;
                font-size: 1.1rem;
                font-weight: 600;
                color: #d97706;
            }

            .warning-box p {
                margin: 0;
                font-size: 0.95rem;
                line-height: 1.5;
            }

            .warning-box hr {
                border: none;
                height: 1px;
                background-color: rgba(146, 64, 14, 0.2); /* A semi-transparent version of the text color */
                margin: 1rem 0;
            }

            .warning-box a {
                color: #b45309; /* A darker shade of the warning text color */
                font-weight: 600;
                text-decoration: none;
            }
            
            .warning-box a:hover {
                text-decoration: underline;
            }


            .chat-container {
                max-height: 600px;
                overflow-y: auto;
                border: 2px solid var(--border-color);
                border-radius: var(--radius-lg);
                padding: 1.5rem;
                background: var(--secondary-color);
                margin-bottom: 2rem;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
                position: relative;
            }

            .chat-container::-webkit-scrollbar {
                width: 8px;
            }

            .chat-container::-webkit-scrollbar-track {
                background: #f1f5f9;
                border-radius: 4px;
            }

            .chat-container::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 4px;
            }

            .chat-container::-webkit-scrollbar-thumb:hover {
                background: #94a3b8;
            }

            .chat-welcome {
                text-align: center;
                padding: 1rem 0.5rem;
                color: var(--text-secondary);
            }

            .chat-welcome-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
                display: block;
            }

            .chat-welcome-text {
                font-size: 1.1rem;
                margin: 0;
            }

            .message {
                margin-bottom: 1.5rem;
                padding: 1rem 1.25rem;
                border-radius: var(--radius-lg);
                max-width: 85%;
                word-wrap: break-word;
                animation: slideIn 0.3s ease-out;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .message-author {
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: var(--text-secondary);
            }

            .user-message .message-author {
                color: rgba(255, 255, 255, 0.9);
            }

            .user-message {
                background: linear-gradient(135deg, #3b82f6 0%, var(--primary-color) 100%); /* More subtle gradient */
                color: white;
                margin-left: auto;
                border-bottom-right-radius: var(--radius-sm);
                box-shadow: var(--shadow-md);
            }

            .ai-message {
                background: #f1f5f9; /* Light gray to distinguish from the container */
                color: var(--text-primary);
                border-bottom-left-radius: var(--radius-sm);
                box-shadow: var(--shadow-sm);
                position: relative; /* For positioning the copy button */
            }

            .ai-message .message-author {
                color: var(--primary-color);
            }

            .ai-message h1,
            .ai-message h2,
            .ai-message h3,
            .ai-message h4,
            .ai-message h5,
            .ai-message h6 {
                color: var(--primary-color);
                margin-top: 0;
                margin-bottom: 0.75rem;
            }

            .ai-message ul,
            .ai-message ol {
                padding-left: 1.5rem;
                margin: 0.75rem 0;
            }

            .ai-message li {
                margin-bottom: 0.5rem;
            }

            .ai-message strong {
                color: var(--primary-color);
            }

            .ai-message p {
                margin: 0.75rem 0;
                line-height: 1.6;
            }

            .ai-message p:first-child {
                margin-top: 0;
            }

            .ai-message p:last-child {
                margin-bottom: 0;
            }

            .timestamp {
                font-size: 0.75rem;
                opacity: 0.7;
                margin-top: 0.5rem;
                font-style: italic;
            }

            .user-message .timestamp {
                color: rgba(255, 255, 255, 0.8);
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); /* Improves readability on gradient */
            }

            .copy-btn {
                position: absolute;
                top: 0.75rem;
                right: 0.75rem;
                background: var(--secondary-color);
                border: 1px solid var(--border-color);
                color: var(--text-secondary);
                border-radius: var(--radius-sm);
                cursor: pointer;
                font-size: 0.9rem;
                padding: 0.25rem 0.5rem;
                opacity: 0; /* Hidden by default, appears on hover */
                transition: opacity 0.2s ease, background-color 0.2s ease, color 0.2s ease;
                line-height: 1;
                z-index: 10;
            }

            .ai-message:hover .copy-btn {
                opacity: 1; /* Show on hover of the message */
            }

            .copy-btn:hover {
                background: #e2e8f0; /* Corresponds to btn-secondary:hover */
                color: var(--text-primary);
            }

            .copy-btn.copied {
                background-color: var(--accent-color);
                border-color: var(--accent-color);
                color: white;
                opacity: 1;
            }

            .input-section {
                display: flex;
                gap: 1rem;
                align-items: flex-start;
            }

            .input-wrapper {
                flex: 1;
            }

            .input-wrapper textarea {
                width: 100%;
                padding: 1rem;
                border: 2px solid var(--border-color);
                border-radius: var(--radius-md);
                font-family: inherit;
                font-size: 1rem;
                resize: vertical;
                min-height: 80px;
                background: white;
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }

            .input-wrapper textarea:focus {
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }

            .input-wrapper textarea::placeholder {
                color: var(--text-muted);
            }

            .button-group {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }

            .btn {
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: var(--radius-md);
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                min-width: 100px;
            }

            .btn-primary {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
                color: white;
                box-shadow: var(--shadow-sm);
            }

            .btn-primary:hover {
                transform: translateY(-1px);
                box-shadow: var(--shadow-md);
            }

            .btn-primary:active {
                transform: translateY(0);
            }

            .btn-secondary {
                background: #f1f5f9;
                color: var(--text-secondary);
                border: 1px solid var(--border-color);
            }

            .btn-secondary:hover {
                background: #e2e8f0;
                color: var(--text-primary);
            }

            .status-indicator {
                margin-top: 1rem;
                padding: 0.75rem 1rem;
                border-radius: var(--radius-md);
                font-size: 0.95rem;
                font-weight: 500;
                text-align: center;
                transition: all 0.3s ease;
            }

            .status-ready {
                background: #f0fdf4;
                color: #166534;
                border: 1px solid #bbf7d0;
            }

            .status-thinking {
                background: #fef3c7;
                color: #92400e;
                border: 1px solid #fbbf24;
            }

            .status-error {
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fca5a5;
            }

            .app-footer {
                text-align: center;
                margin-top: 2rem;
                padding-top: 1.5rem;
                border-top: 1px solid var(--border-color);
                color: var(--text-muted);
                font-size: 0.9rem;
            }

            .divider {
                height: 1px;
                background: linear-gradient(to right, transparent, var(--border-color), transparent);
                margin: 2rem 0;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .main-container {
                    padding: 1rem;
                    margin: 1rem;
                }
                
                .app-title {
                    font-size: 2rem;
                }
                
                .chat-container {
                    max-height: 400px;
                }
                
                .message {
                    max-width: 95%;
                }
                
                .input-section {
                    flex-direction: column;
                }
                
                .button-group {
                    flex-direction: row;
                    width: 100%;
                }
                
                .btn {
                    flex: 1;
                }
            }

            /* Spinner Animation */
            .typing-indicator {
                display: inline-block;
                margin-right: 0.5rem;
                vertical-align: middle;
            }
            .typing-indicator span {
                height: 8px;
                width: 8px;
                background-color: var(--text-secondary);
                border-radius: 50%;
                display: inline-block;
                margin: 0 2px;
                animation: typing-bounce 1.4s infinite both;
            }
            .typing-indicator span:nth-child(1) {
                animation-delay: -0.32s;
            }
            .typing-indicator span:nth-child(2) {
                animation-delay: -0.16s;
            }
            @keyframes typing-bounce {
                0%, 80%, 100% {
                    transform: scale(0);
                } 40% {
                    transform: scale(1.0);
                }
            }

            /* Loading animation */
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                }
                50% {
                    opacity: 0.5;
                }
            }

            .loading {
                animation: pulse 1.5s ease-in-out infinite;
            }

            /* Smooth transitions */
            .fade-in {
                animation: fadeIn 0.5s ease-out;
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Two-column Layout */
            .content-wrapper {
                display: flex;
                gap: 2rem;
                align-items: flex-start;
                margin-top: 2rem;
            }

            .sidebar-column {
                flex: 0 0 320px; /* Fixed width for the sidebar */
                position: sticky; /* Make it stick on scroll */
                top: 2rem;
            }

            .chat-column {
                flex: 1; /* Take up remaining space */
                min-width: 0; /* Prevent flexbox overflow */
            }

            /* Responsive stacking for smaller screens */
            @media (max-width: 1024px) {
                .content-wrapper {
                    flex-direction: column;
                }
                .sidebar-column {
                    position: static; /* Unstick it on smaller screens */
                    width: 100%;
                }
            }
        """),
        ui.tags.script("""
            // This function sets up all the interactive features on the page.
            function initializeInteractions() {
                const chatContainer = document.getElementById('chat-container');
                const userInput = document.getElementById('user_input');
                const sendButton = document.getElementById('send_message');

                // --- Feature 1: Ctrl+Enter to send message ---
                if (userInput && sendButton) {
                    userInput.addEventListener('keydown', function(event) {
                        if (event.key === 'Enter' && event.ctrlKey) {
                            event.preventDefault(); // Prevent new line in textarea
                            sendButton.click(); // Programmatically click the send button
                        }
                    });
                }

                // --- Feature 2: Copy to Clipboard for AI messages ---
                if (chatContainer) {
                    chatContainer.addEventListener('click', function(event) {
                        const copyButton = event.target.closest('.copy-btn');
                        if (copyButton) {
                            const textToCopy = copyButton.getAttribute('data-copy-text');
                            navigator.clipboard.writeText(textToCopy).then(() => {
                                // Provide feedback to the user
                                copyButton.classList.add('copied');
                                copyButton.textContent = 'Copied!';
                                setTimeout(() => {
                                    copyButton.innerHTML = '&#128203;'; // Revert to icon
                                    copyButton.classList.remove('copied');
                                }, 2000);
                            }).catch(err => {
                                console.error('Failed to copy text: ', err);
                            });
                        }
                    });
                }
            }

            // --- Auto-scrolling logic (from your original script) ---
            function scrollChatToBottom() {
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    setTimeout(() => { chatContainer.scrollTop = chatContainer.scrollHeight; }, 100);
                }
            }

            // Initialize everything when the document is ready.
            document.addEventListener('DOMContentLoaded', function() {
                initializeInteractions();
                const observer = new MutationObserver(scrollChatToBottom);
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    observer.observe(chatContainer, { childList: true, subtree: true });
                }
            });
        """)
    ),
    ui.div(
        # Header section from the new stylesheet
        ui.div(
            ui.h1("🏥 Dokter Arki", class_="app-title"),
            ui.p("Your Artificial Intelligence Medical Assistant", class_="app-subtitle"),
            class_="app-header"
        ),
        
        # Main content area with two columns
        ui.div(
            # Left column for disclaimer
            ui.div(
                ui.div(
                    ui.h5("⚠️ Important Medical Disclaimer"),
                    ui.p("Asisten AI ini dibuat hanya untuk tujuan pendidikan dan informasi. Bukan sebagai pengganti nasihat medis profesional, diagnosis, atau pengobatan. Selalu konsultasikan dengan tenaga medis yang berkualifikasi untuk masalah kesehatan Anda."),
                    ui.hr(),
                    ui.p(
                        "Chatbot ini didukung oleh ",
                        ui.tags.a(
                            "II-Medical-8B-1706",
                            href="https://huggingface.co/Intelligent-Internet/II-Medical-8B-1706",
                            target="_blank",
                            rel="noopener noreferrer"
                        ),
                        ", model bahasa besar canggih terbaru yang dikembangkan oleh Intelligent Internet, yang dirancang khusus untuk meningkatkan penalaran medis berbasis AI.",
                        style="font-size: 0.9rem; font-style: italic;"
                    ),
                    class_="warning-box"
                ),
                class_="sidebar-column"
            ),
            # Right column for chat interface
            ui.div(
                ui.div(
                    ui.output_ui("render_chat_history"),
                    id="chat-container",
                    class_="chat-container"
                ),
                ui.div(
                    ui.div( # Input wrapper for the textarea
                        ui.input_text_area(
                            "user_input",
                            "",
                            placeholder="Ask a medical question or describe symptoms... (Ctrl+Enter to send)",
                            rows=3,
                            width="100%"
                        ),
                        class_="input-wrapper"
                    ),
                    ui.div( # Button group for the action buttons
                        ui.input_action_button(
                            "send_message",
                            "Send",
                            class_="btn btn-primary",
                        ),
                        ui.input_action_button(
                            "clear_chat",
                            "Clear",
                            class_="btn btn-secondary",
                        ),
                        class_="button-group"
                    ),
                    class_="input-section"
                ),
                ui.div(ui.output_ui("status"), id="status-indicator"),
                class_="chat-column"
            ),
            class_="content-wrapper"
        ),
        
        # Footer
        ui.p(
            "This app is built for educational purposes by Adnuri Mohamidi with help from AI.",
            class_="app-footer"
        ),
        
        class_="main-container"
    )
)

# Server Logic
def server(input, output, session):
    # Initialize clients
    try:
        hf_client = initialize_hf_client()
        print("✅ Hugging Face Client initialized successfully")
    except Exception as e:
        hf_client = None
        print(f"❌ Error initializing Hugging Face client: {e}")

    try:
        google_client = initialize_google_client()
        print("✅ Google AI Client for translation initialized successfully")
    except Exception as e:
        google_client = None
        print(f"❌ Error initializing Google AI client: {e}")
    
    # Reactive values for chat history
    chat_history = reactive.Value([])
    is_processing = reactive.Value(False)
    
    def detect_language(text: str, client: genai.GenerativeModel) -> str:
        """Detects the language of the given text using Google's Gemini model."""
        if not client:
            print("⚠️ Language detection skipped: Google AI client not initialized.")
            return "English"  # Default to English if client is not available

        print(f"🔎 Detecting language for: {text[:50]}...")
        try:
            prompt = f"What language is the following text written in? Respond with only the language name (e.g., 'Indonesian', 'English', 'Spanish').\n\nText: \"{text}\""
            response = client.generate_content(prompt)
            detected_lang = response.text.strip()
            # Handle cases where the model might be verbose
            if '\n' in detected_lang:
                detected_lang = detected_lang.split('\n')[0]
            print(f"✅ Language detected: {detected_lang}")
            return detected_lang
        except Exception as e:
            print(f"❌ Exception during language detection: {e!r}. Defaulting to English.")
            return "English"

    def translate_text(text: str, target_language: str, client: genai.GenerativeModel) -> str:
        """Translates text to the target language using Google's Gemini model."""
        if not client:
            return f"(Translation failed: Google AI client not initialized) {text}"

        # No need to translate if the text is already in the target language
        if target_language.lower() == 'english':
            print("✅ No translation needed (target is English).")
            return text

        print(f"🔄 Translating response to {target_language}...")
        try:
            prompt = f"Translate the following English text to {target_language}. Only provide the translation, nothing else:\n\n{text}"
            response = client.generate_content(prompt)
            print("✅ Translation successful.")
            return response.text
        except Exception as e:
            print(f"❌ Exception during translation to {target_language}: {e!r}")
            return f"(Translation to {target_language} failed, showing original text) {text}"

    def get_ai_response(user_message: str) -> str:
        """Get response from the medical AI model and translate it to the user's language."""
        print(f"🔄 Getting AI response for: {user_message[:50]}...")
        
        if not hf_client:
            print("❌ Hugging Face client not initialized!")
            return "Error: AI client not initialized. Please check your API key configuration."
        
        # 1. Detect the language of the user's input
        user_lang = detect_language(user_message, google_client)
        
        response_content = ""
        try:
            # Create messages for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are Dokter Arki, a helpful medical AI assistant. Provide accurate, educational information. Structure your answer for clarity and readability. Use markdown formatting such as bullet points (`* item`), numbered lists, and bold text (`**text**`) where it helps to explain complex information. Always conclude your response by reminding the user to consult a qualified healthcare professional for any medical advice, diagnosis, or treatment."
                },
                {"role": "user", "content": user_message},
            ]
            
            print("📡 Sending request to Hugging Face API...")
            completion = hf_client.chat.completions.create(
                model="Intelligent-Internet/II-Medical-8B-1706", messages=messages, max_tokens=1000, temperature=0.7
            )
            
            if completion.choices:
                response_content = completion.choices[0].message.content
                print(f"✅ Received English response: {response_content[:100]}...")
            else:
                response_content = "The AI model returned an empty response. Please try rephrasing your question or try again later."
                print("⚠️ API call successful, but the model returned no choices.")
            
        except StopIteration:
            # This can happen if the model fails to generate any output and the client library doesn't handle it gracefully.
            response_content = "The AI model did not generate a response. This might be a temporary issue with the service. Please try again."
            print("❌ StopIteration caught during API call. The model likely produced no output.")
        except Exception as e:
            print(f"❌ Exception during API call: {e!r}")
            # For user-facing errors, it's better to be generic.
            response_content = "An error occurred while communicating with the AI. Please try again later."

        # 2. Translate the final response content (whether it's a success or an error message)
        return translate_text(response_content, user_lang, google_client)
    
    @reactive.Effect
    @reactive.event(input.send_message)
    def handle_send_message():
        print("🎯 Send message triggered")
        
        try:
            user_message = input.user_input().strip()
            print(f"📝 User message: {user_message}")
            
            if not user_message:
                print("⚠️ No user message provided.")
                return
            
            # Clear input immediately
            ui.update_text_area("user_input", value="")
            
            # Add user message to chat history
            current_history = chat_history.get()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Add user message
            updated_history = current_history + [{
                "type": "user",
                "content": user_message,
                "timestamp": timestamp
            }]
            
            chat_history.set(updated_history)
            print("📝 Added user message to history")
            
            # Set processing state
            is_processing.set(True)
            
            # Get AI response (synchronous)
            ai_response = get_ai_response(user_message)
            
            # Add AI response to chat history
            final_history = updated_history + [{
                "type": "ai",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }]
            
            chat_history.set(final_history)
            is_processing.set(False)
            
            print("✅ Message handling completed successfully")
            
        except Exception as e:
            print(f"❌ Exception in handle_send_message: {e}")
            is_processing.set(False)
    
    @reactive.Effect
    @reactive.event(input.clear_chat)
    def handle_clear_chat():
        print("🗑️ Clearing chat history")
        chat_history.set([])
    
    @output
    @render.ui
    def render_chat_history():
        history = chat_history.get()
        
        if not history:
            # Use the new, more engaging welcome message style
            return ui.div(
                ui.tags.span("👋", class_="chat-welcome-icon"),
                ui.p(
                    "Selamat datang! Tanyakan kepada saya pertanyaan medis apa pun atau jelaskan gejala yang Anda alami. Gunakan Bahasa Indonesia, Inggris atau lainnya, saya siap menjawabnya!",
                    class_="chat-welcome-text"
                ),
                class_="chat-welcome"
            )
        
        messages = []
        for i, msg in enumerate(history):
            if msg["type"] == "user":
                messages.append(
                    ui.div(
                        ui.div("You", class_="message-author"),
                        ui.div(msg["content"]),
                        ui.div(msg["timestamp"], class_="timestamp"),
                        class_="message user-message",
                        id=f"msg-{i}"
                    )
                )
            else:
                messages.append(
                    ui.div(
                        # Add a copy button that the JS in main.js will use
                        ui.div("AI Assistant", class_="message-author"),
                        ui.tags.button(
                            "📋",
                            class_="copy-btn",
                            title="Copy response to clipboard",
                            **{"data-copy-text": msg["content"]},
                        ),
                        ui.markdown(msg["content"]),
                        ui.div(msg["timestamp"], class_="timestamp"),
                        class_="message ai-message",
                        id=f"msg-{i}"
                    )
                )
        
        return ui.div(*messages)
    
    @output
    @render.ui
    def status():
        status_class = "status-indicator"
        if is_processing.get():
            status_class += " status-thinking"
            return ui.div(
                ui.div(
                    ui.tags.span(), ui.tags.span(), ui.tags.span(), class_="typing-indicator"
                ),
                "AI is thinking...",
                class_=status_class
            )
        elif not hf_client:
            status_text = "❌ Error: Medical AI client not initialized"
            status_class += " status-error"
        elif not google_client:
            status_text = "❌ Error: Translation client not initialized"
            status_class += " status-error"
        else:
            status_text = "✅ Ready to help with medical questions"
            status_class += " status-ready"
        
        return ui.div(status_text, class_=status_class)


# Create the app
app = App(app_ui, server)

# This block is essential for running the app locally using `python app.py`.
if __name__ == "__main__":
    print("🚀 Starting Medical Chatbot...")
    app.run()
