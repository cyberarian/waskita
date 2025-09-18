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
    ui.tags.head(
        ui.tags.style("""
            .chat-container {
                max-height: 500px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            .message {
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 8px;
            }
            .user-message {
                background-color: #007bff;
                color: white;
                text-align: right;
            }
            .ai-message {
                background-color: #e9ecef;
                color: #333;
            }
            .timestamp {
                font-size: 0.8em;
                opacity: 0.7;
                margin-top: 5px;
            }
            .warning-box {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                color: #856404;
            }
        """)
    ),
    ui.tags.script("""
        // Function to scroll the chat container to the bottom
        function scrollChatToBottom() {
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                setTimeout(() => {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }, 100);
            }
        }

        // Use a MutationObserver to watch for changes in the chat container
        const observer = new MutationObserver(scrollChatToBottom);
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                observer.observe(chatContainer, { childList: true, subtree: true });
            }
        });
    """),
    ui.div(
        ui.h1("🏥 Waskita Husada", class_="text-center"),
        ui.p("Your AI Medical Assistant", class_="text-center text-muted mb-4"),
        
        # Warning disclaimer
        ui.div(
            ui.h5("⚠️ Important Medical Disclaimer"),
            ui.p("This AI assistant is for educational and informational purposes only. "
                 "It should not be used as a substitute for professional medical advice, "
                 "diagnosis, or treatment. Always consult with qualified healthcare "
                 "professionals for medical concerns."),
            class_="warning-box"
        ),
        
        # Chat interface
        ui.div(
            ui.output_ui("render_chat_history"),
            class_="chat-container",
            id="chat-container"
        ),
        
        # Input section
        ui.row(
            ui.column(10,
                ui.input_text_area(
                    "user_input",
                    "",
                    placeholder="Ask a medical question or describe symptoms...",
                    rows=3,
                    width="100%"
                )
            ),
            ui.column(2,
                ui.input_action_button(
                    "send_message",
                    "Send",
                    class_="btn-primary",
                    style="margin-top: 10px; width: 100%;"
                ),
                ui.input_action_button(
                    "clear_chat",
                    "Clear",
                    class_="btn-secondary",
                    style="margin-top: 10px; width: 100%;"
                )
            )
        ),
        
        # Status indicator
        ui.div(
            ui.output_text("status"),
            class_="mt-3 text-muted"
        ),
        
        # Footer
        ui.hr(),
        ui.p(
            "This app is built for educational purposes by Adnuri Mohamidi with help from AI.",
            class_="text-center text-muted small"
        ),
        
        class_="container-fluid",
        style="max-width: 1200px; margin: 0 auto; padding: 20px;"
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
                    "content": "You are a helpful medical AI assistant. Provide accurate, educational information. Structure your answer for clarity and readability. Use markdown formatting such as bullet points (`* item`), numbered lists, and bold text (`**text**`) where it helps to explain complex information. Always conclude your response by reminding the user to consult a qualified healthcare professional for any medical advice, diagnosis, or treatment."
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
            return ui.div(
                ui.p("👋 Welcome! Ask me any medical question or describe your symptoms.", 
                     class_="text-center text-muted"),
                style="padding: 50px;"
            )
        
        messages = []
        for i, msg in enumerate(history):
            if msg["type"] == "user":
                messages.append(
                    ui.div(
                        ui.div(msg["content"]),
                        ui.div(f"You - {msg['timestamp']}", class_="timestamp"),
                        class_="message user-message",
                        id=f"msg-{i}"
                    )
                )
            else:
                messages.append(
                    ui.div(
                        ui.markdown(msg["content"]),
                        ui.div(f"AI Assistant - {msg['timestamp']}", class_="timestamp"),
                        class_="message ai-message",
                        id=f"msg-{i}"
                    )
                )
        
        return ui.div(*messages)
    
    @output
    @render.text
    def status():
        if is_processing.get():
            return "🤔 AI is thinking..."
        elif not hf_client:
            return "❌ Error: Medical AI client not initialized"
        elif not google_client:
            return "❌ Error: Translation client not initialized"
        else:
            return "✅ Ready to help with medical questions"

# Create the app
app = App(app_ui, server)

# This block is essential for running the app locally using `python app.py`.
if __name__ == "__main__":
    print("🚀 Starting Medical Chatbot...")
    app.run(debug=True)
