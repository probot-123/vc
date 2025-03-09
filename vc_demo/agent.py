import asyncio
import logging
import json
import openai
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, silero, turn_detector

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

# Store conversation history
conversation_log = []

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def attach_user_final_handler(agent: VoicePipelineAgent, handler):
    # Wait until agent._human_input is available, then attach the handler.
    while getattr(agent, "_human_input", None) is None:
        await asyncio.sleep(0.5)
    agent._human_input.on("final_transcript", handler)
    logger.info("Attached user final_transcript handler to human_input")

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoid using unpronounceable punctuation. "
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Configure Voice Assistant
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=openai.stt.STT(language="en", model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    # Handler for agent speech committed events.
    @agent.on("agent_speech_committed")
    def handle_agent_transcript(data):
        # data is a ChatMessage. Its content holds the agent's reply.
        content = data.content
        transcript = ""
        if isinstance(content, str):
            transcript = content.strip()
        elif isinstance(content, list) and content:
            # Assume the first element is the text (if it's a string)
            if isinstance(content[0], str):
                transcript = content[0].strip()
        # ChatMessage does not provide a timestamp by default.
        timestamp = ""
        if transcript:
            log_entry = {"role": "assistant", "timestamp": timestamp, "text": transcript}
            conversation_log.append(log_entry)
            logger.info(f"Assistant said: {transcript}")

    @agent.on("agent_speech_interrupted")
    def handle_agent_interrupted(data):
        content = data.content
        transcript = ""
        if isinstance(content, str):
            transcript = content.strip()
        elif isinstance(content, list) and content:
            if isinstance(content[0], str):
                transcript = content[0].strip()
        timestamp = ""
        if transcript:
            log_entry = {"role": "assistant", "timestamp": timestamp, "text": transcript + " [interrupted]"}
            conversation_log.append(log_entry)
            logger.info(f"Assistant interrupted: {transcript}")

    # Handler for user final transcript events.
    def handle_final_transcript(data):
        # data is expected to be a SpeechEvent with an 'alternatives' attribute.
        transcript = ""
        if hasattr(data, "alternatives") and data.alternatives:
            alt = data.alternatives[0]
            transcript = alt.text.strip() if hasattr(alt, "text") and alt.text else ""
        # User messages emitted via HumanInput don't include a timestamp.
        timestamp = ""
        if transcript:
            enhanced_text = enhance_transcript(transcript)
            log_entry = {"role": "user", "timestamp": timestamp, "text": enhanced_text}
            conversation_log.append(log_entry)
            logger.info(f"User said: {enhanced_text}")

    # Attach the user final transcript handler once human_input is available.
    asyncio.create_task(attach_user_final_handler(agent, handle_final_transcript))

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    # Start the voice agent
    agent.start(ctx.room, participant)

    # Greet the user
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

    # Loop indefinitely (since wait_until_done is not available)
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        save_transcript()  # Save conversation when session ends

def enhance_transcript(text):
    """Enhances transcripts for readability using OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Enhance the transcript to be more readable and natural."},
                {"role": "user", "content": text},
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error enhancing transcript: {e}")
        return text

def save_transcript():
    """Saves the conversation history to a JSON file."""
    try:
        with open("conversation_log.json", "w") as f:
            json.dump(conversation_log, f, indent=4)
        logger.info("Conversation saved to conversation_log.json")
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
