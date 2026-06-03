# LangGraph Football Scout AI

An AI-powered football scouting application that fetches live player data from TheSportsDB API and generates detailed, position-specific scouting reports using Groq's Llama 3.3 70B model — all orchestrated through a LangGraph state machine.

## How it works

1. You type a player or manager name
2. The app fetches live data from TheSportsDB (club, position, nationality, date of birth)
3. LangGraph classifies the position and routes to the correct scout node
4. A position-specific prompt generates a structured scouting report with attribute ratings out of 10, a speciality label, and an overall conclusion

## Position routing

| Position | Scout focus |
|---|---|
| Forward | Goals, xG, pressing, finishing technique |
| Midfielder | Passing range, press resistance, creativity, spatial awareness |
| Defender | Tackling, aerial duels, build-up play, positional discipline |
| Goalkeeper | Shot stopping, command of area, distribution, sweeper ability, penalties |
| Manager | Formation, press intensity, style, rotation, big game record |

## Tech stack

- **LangGraph** — state machine graph orchestration
- **LangChain Core** — prompt templates
- **Groq / Llama 3.3 70B** — LLM inference
- **Streamlit** — web UI
- **TheSportsDB API** — live player data
- **python-dotenv** — environment variable management

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install langgraph langchain-core langchain-groq streamlit requests python-dotenv
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Or run the CLI version directly:
```bash
python scout.py
```
