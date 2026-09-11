import os

import websockets

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
# reuse the same host:port as the REST client, just swap the scheme
WS_URL = BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://")


async def stream_chat(session_id: str, message: str):
    uri = f"{WS_URL}/chat/ws/{session_id}"

    async with websockets.connect(uri) as ws:
        await ws.send(message)

        async for msg in ws:
            if msg == "[DONE]":
                break
            yield msg
