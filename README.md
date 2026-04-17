# Engineering Companion

A production-minded Streamlit app that serves as a structured teaching system for engineering, BIM, Revit, construction workflows, technical standards, software learning, and engineering documentation. It transforms user-provided educational content into an interactive, organized learning experience with adaptive depth, background awareness, and multiple display modes, tailored for engineering professionals and learners.

## Features

- **Engineering-Focused Teaching**: Specialized in BIM, Revit, construction, technical standards, software, and documentation.
- **Language-First Flow**: Choose teaching language (Arabic or English) before starting.
- **Learning Depth Selection**: Quick Overview, Guided Learning, Deep Learning, or Expert Refresh.
- **Background-Aware Teaching**: Adapts to Beginner, Some Familiarity, Already Know Basics, or Advanced levels.
- **Display Modes**: Presentation View, Book View, Slide-by-Slide View, or Scroll View.
- **Subject Storage & Resume**: Save subjects locally and resume previously created subjects.
- **OpenRouter Integration**: Uses OpenRouter for AI-powered explanations with free model filtering.
- **Streaming Responses**: Buffered streaming output for teaching content.
- **Export Options**: Export lesson content as TXT or JSON.
- **Clean Minimal UI**: Professional, educational interface inspired by Visual Studio Code.

## What Makes It Different

This is **not a generic chatbot**. It's a disciplined teaching system for engineering with:
- Strict educational protocol for progressive explanations in engineering contexts.
- State-based flow ensuring structured interactions.
- Multi-layered prompting for consistent pedagogy.
- Adaptation to user settings without generic responses.
- Focus on building understanding step-by-step in engineering applications, not answering arbitrary questions.

## Supported Languages

- **UI Language**: English or Arabic (full interface localization).
- **Teaching Language**: Arabic (natural Egyptian style) or English (professional), separate from UI language.
- **Source Material**: Any language; teaching adapts accordingly.

## How to Run

1. Clone or download this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Open the provided URL in your browser.

## Getting Started

1. **Set UI Language**: Choose English or Arabic for the interface.
2. **Choose Teaching Language**: Select Arabic or English for explanations.
3. **Select Learning Depth**: Pick the appropriate depth for your engineering needs.
4. **Set Background Level**: Indicate your prior engineering knowledge.
5. **Pick Display Mode**: Choose how you want to view the content.
6. **Add Material**: Paste text or upload a .txt/.md file with engineering content.
7. **Specify Parts**: Decide how many parts to divide the explanation into.
8. **Learn**: Navigate through parts, use commands like "next", "deeper", "summarize".
9. **Resume Later**: Subjects are saved locally and can be resumed from the sidebar.

## OpenRouter Setup

- Visit [OpenRouter Keys](https://openrouter.ai/keys) to get your API key.
- Enter the key in the app's Settings panel.
- The app automatically filters for free models only.

## Usage Tips

- **Commands**: Use "next", "2", "deeper", "simpler", "summarize", "recap" during teaching.
- **Arabic Teaching**: Explanations use natural Egyptian Arabic for clarity and approachability.
- **Depth Adaptation**: Content density and detail adjust based on your selected depth.
- **Background Adaptation**: Explanations pace and terminology adapt to your knowledge level.
- **Display Modes**: Switch modes mid-session for different viewing experiences.
- **Export**: Save your learning sessions for review or sharing.

## Limitations

- Current version supports pasted text and .txt/.md uploads only.
- Free OpenRouter models only; no paid model support.
- Local JSON storage for subjects and progress; no cloud sync.
- The interface is focused on a lightweight teaching flow; advanced image or file-type support is not implemented.

## Future Improvements

- Support for more file types (PDF, DOCX, images).
- Additional languages.
- Cloud storage integration.
- Quiz generation and assessment.
- Collaborative features.
- Advanced image and diagram support.

## Architecture Notes

- Built with Streamlit for rapid UI development.
- State-based flow using session state.
- Modular prompting layers for teaching consistency.
- Lightweight local JSON storage for subjects.
- Extensible provider architecture (currently OpenRouter-focused).

## Contributing

This is a structured educational tool. Contributions should maintain the teaching protocol and UI professionalism.

## License

[Add your license here, e.g., MIT]

---

**Instructor Companion**: Transforming content into structured learning experiences.